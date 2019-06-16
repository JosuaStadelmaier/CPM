import tensorflow as tf
import numpy as np
from TransE import TransE
from data_processing import DataProcessing as Data

'''
This class handles training and evaluation of 'plain' KBC models.
Structure:
- Data for training and evaluation
- Training
- Evaluation
'''


class KBC(TransE):

    def __init__(self, config, args):
        super().__init__(config, args)

        if not self.args.evaluate:
            self.training_data()
            self.evaluation_data()

            self.training_loss = self.loss(self.training_data_batches)
            learning_rate = self.config['learning_rate'] if not self.args.paths else self.config['path_learning_rate']
            self.optimize = self.optimization(learning_rate, self.config['optimizer'])\
                .minimize(self.training_loss, global_step=self.global_step)

            with tf.control_dependencies([self.entity_embedding]):
                self.initial_normalizer = self.entity_embedding.assign(tf.nn.l2_normalize(self.entity_embedding, axis=-1))

            with tf.control_dependencies([self.optimize]):
                self.normalize = self.entity_embedding.assign(tf.nn.l2_normalize(self.entity_embedding, axis=-1))

            self.validation_loss = self.loss({1: self.validation_batch})
            self.train_scores = {l: self.evaluation(datapoint) for l, datapoint in self.train_evaluation_datapoints.items()}
            self.valid_scores = {l: self.evaluation(datapoint) for l, datapoint in self.valid_evaluation_datapoints.items()}

            self.saver = tf.train.Saver(save_relative_paths=True)

    '''
    DATA FOR TRAINING AND EVALUATION
    '''

    def training_data(self):
        self.training_data_iterators = []
        self.training_data_batches = {}
        path_lengths = self.config['path_lengths'] if self.args.paths else [1]
        for path_length in path_lengths:
            with tf.name_scope('path_length_{}'.format(path_length)):
                training_iterator, training_batch = self.training_data_with_paths(path_length)
                self.training_data_iterators.append(training_iterator)
                self.training_data_batches[path_length] = training_batch

    def training_data_with_paths(self, path_length):
        with tf.name_scope('data'):
            # data set for training
            column_types = [tf.constant([0], dtype=tf.int32, shape=[])] * (3+2*self.config['neg_samples']+path_length)
            # in path training, each batch consists of smaller, equally sized batches of the different path lengths
            batch_size = self.config['batch_size']//len(list(self.config['path_lengths'])) if self.args.paths \
                else self.config['batch_size']
            file_name = 'path_training_file'
            path_training_file = self.config[file_name][path_length]
            training_data = tf.data.experimental.CsvDataset(path_training_file, column_types) \
                .apply(tf.data.experimental.shuffle_and_repeat(self.config['shuffle_buffer_size']))\
                .batch(batch_size).prefetch(1)
            # .cache().apply
            training_iterator = training_data.make_initializable_iterator()
            # convert to tensor with shape: (batch size, line width)
            training_batch = tf.transpose(tf.convert_to_tensor(training_iterator.get_next()), perm=[1, 0])
            return training_iterator, training_batch

    def evaluation_data(self):
        with tf.name_scope('evaluation_data'):
            # data set for computing the validation loss on single edges
            # output: [e1/e2/both corrupted, e1, e2] + e1_c + e2_c + r
            column_types = [tf.constant([0], dtype=tf.int32, shape=[])] * (3+2*self.config['neg_samples']+1)
            file_name = 'valid_file_neg_samples'
            validation_loss_data = tf.data.experimental.CsvDataset([self.config[file_name]], column_types)\
                .batch(self.config['batch_size']).prefetch(1)
            self.validation_loss_iterator = validation_loss_data.make_initializable_iterator()
            validation_batch = self.validation_loss_iterator.get_next()
            # convert to tensor with shape: (batch size, line width)
            self.validation_batch = tf.transpose(tf.convert_to_tensor(validation_batch), perm=[1, 0])

            # data sets for computing ranking metrics
            self.train_evaluation_iterators = []
            self.train_evaluation_datapoints = {}
            self.valid_evaluation_iterators = []
            self.valid_evaluation_datapoints = {}

            for path_length in [1, 2, 3]:
                train_iterator, train_datapoint = self.ranking_evaluation_data('path_train_file_neg_domain', path_length)
                self.train_evaluation_iterators.append(train_iterator)
                self.train_evaluation_datapoints[path_length] = train_datapoint

                valid_iterator, valid_datapoint = self.ranking_evaluation_data('path_valid_file_neg_domain', path_length)
                self.valid_evaluation_iterators.append(valid_iterator)
                self.valid_evaluation_datapoints[path_length] = valid_datapoint

    def untyped_evaluation_data(self, file_path):
        possible_entities = set(self.index.entity_index['id_to_name'].keys())
        original_generator = Data.evaluation_data_reader(file_path)
        for e1, e2, e1_domain, e2_domain, path in original_generator():
            e1_c_domain = list(possible_entities - set(e1_domain))
            e2_c_domain = list(possible_entities - set(e2_domain))
            yield e1, e2, e1_c_domain, e2_c_domain, path

    def ranking_evaluation_data(self, file_name, path_length):
        # output: (e1, e2, e1_domain, e2_domain, path)
        output_types = (tf.int32,) * 5
        output_shapes = (tf.TensorShape([]),) * 2 + (tf.TensorShape([None]),) * 3
        file_path = self.config[file_name][path_length]
        if self.config['type_matching']:
            generator = Data.evaluation_data_reader(file_path)
        else:
            generator = lambda: self.untyped_evaluation_data(file_path)
        evaluation_data = tf.data.Dataset.from_generator(generator, output_types, output_shapes).prefetch(1)
        iterator = evaluation_data.make_initializable_iterator()
        datapoint = iterator.get_next()
        return iterator, datapoint

    def parse_data_set(self, data_set):
        # expand dimensions to enable broadcasting in self.predict()
        e1 = tf.expand_dims(data_set[:, 1], 1)
        e2 = tf.expand_dims(data_set[:, 2], 1)
        e1_c = data_set[:, 3:3 + self.config['neg_samples']]
        e2_c = data_set[:, 3 + self.config['neg_samples']:3 + 2 * self.config['neg_samples']]
        path = tf.expand_dims(data_set[:, 3 + 2 * self.config['neg_samples']:], 2)
        return e1, e2, e1_c, e2_c, path

    '''
    TRAINING
    '''

    def optimization(self, learning_rate, algorithm):
        optimizers = {'Adagrad': tf.train.AdagradOptimizer,
                      'Adadelta': tf.train.AdadeltaOptimizer,
                      'Adam': tf.train.AdamOptimizer}

        optimizer = optimizers[algorithm](learning_rate)

        return optimizer

    def loss(self, data_sets):
        loss_batches = []
        for path_length, data_batch in data_sets.items():
            with tf.name_scope('loss_path_length_{}'.format(path_length)):

                # separate batches by the availability of corrupted entities
                # by using the first column of data_set: 0 => only e1 corrupted, 1 => only e2 corrupted, 2 => both corrupted
                e1_c_batch = tf.boolean_mask(data_batch, tf.not_equal(data_batch[:, 0], 1))
                e2_c_batch = tf.boolean_mask(data_batch, tf.greater_equal(data_batch[:, 0], 1))

                # e2 gets corrupted
                e1, e2, e1_c, e2_c, path = self.parse_data_set(e2_c_batch)

                correct_similarity_shifted = self.predict(e1, path, e2, sigmoid=False)
                labels_true = tf.ones_like(correct_similarity_shifted)
                e2_correct_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_true, logits=correct_similarity_shifted)
                e2_correct_loss = tf.reshape(e2_correct_loss, [-1])  # flatten for concat

                corrupted_similarity_shifted = self.predict(e1, path, e2_c, sigmoid=False)
                labels_false = tf.zeros_like(corrupted_similarity_shifted)
                e2_corrupted_loss =\
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_false, logits=corrupted_similarity_shifted)/10.0

                e2_corrupted_loss = tf.reshape(e2_corrupted_loss, [-1])  # flatten for concat
                loss_batches.extend([e2_correct_loss, e2_corrupted_loss])

                # e1 gets corrupted
                e1, e2, e1_c, e2_c, path = self.parse_data_set(e1_c_batch)

                correct_similarity_shifted = self.predict(e1, path, e2, sigmoid=False)
                labels_true = tf.ones_like(correct_similarity_shifted)
                e1_correct_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_true,
                                                                       logits=correct_similarity_shifted)
                e1_correct_loss = tf.reshape(e1_correct_loss, [-1])  # flatten for concat

                corrupted_similarity_shifted = self.predict(e1_c, path, e2, sigmoid=False)
                labels_false = tf.zeros_like(corrupted_similarity_shifted)
                e1_corrupted_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_false,
                                                                         logits=corrupted_similarity_shifted)/10.0

                e1_corrupted_loss = tf.reshape(e1_corrupted_loss, [-1])  # flatten for concat
                loss_batches.extend([e1_correct_loss, e1_corrupted_loss])

        loss = tf.reduce_mean(tf.concat(loss_batches, axis=0))
        return loss

    def train(self):
        summary_description = self.args.d + '/' if self.args.d else ''
        writer = tf.summary.FileWriter(self.config['summary_dir'] + '/' + summary_description + self.describe())
        path_prefix = 'path_' if self.args.paths else ''
        description = 'c{}_{}'.format(self.config['id'], self.args.d) if self.args.d else self.describe()
        checkpoint_dir = self.config['checkpoint_dir'] + '/' + self.config['model'] + '/'
        checkpoint_path = checkpoint_dir + path_prefix + description + '/model-at-step'
        config = tf.ConfigProto(inter_op_parallelism_threads=12, intra_op_parallelism_threads=12)
        with tf.Session(config=config) as sess:
            # Data initialization
            sess.run([iterator.initializer for iterator in self.training_data_iterators])

            # Restoring from checkpoint
            if self.args.restore or self.args.paths:
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + path_prefix + description)
                if latest_checkpoint is None:
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + description)
                print("Restore model from {}".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)

            # Variable initialization and initial normalization
            else:
                sess.run(tf.global_variables_initializer())
                if self.config['normalize']:
                    sess.run(self.initial_normalizer)

            # Training
            total_loss = 0.0
            for local_step in range(self.config['training_steps'] + 1):
                step = sess.run(self.global_step)

                # Optimization and normalization of entity embeddings
                if self.config['normalize']:
                    loss_batch, _, _ = sess.run([self.training_loss, self.optimize, self.normalize])
                else:
                    loss_batch, _ = sess.run([self.training_loss, self.optimize])

                # Evaluation reports
                total_loss += loss_batch
                if (local_step % self.config['skip_steps'] == 0 or local_step == self.config['training_steps']) and local_step > 0:
                    print("\nstep: {}".format(step))

                    training_loss = total_loss / self.config['skip_steps']
                    total_loss = 0.0
                    if step % self.config['skip_steps'] == 0 and local_step > 0:
                        self.add_summaries(writer, step, [self.custom_summary("training loss", training_loss)])

                    sess.run([self.validation_loss_iterator.initializer]
                             + [i.initializer for i in self.train_evaluation_iterators + self.valid_evaluation_iterators])

                    validation_loss_summary = self.validate_loss(sess)
                    summaries = []
                    if not self.args.paths and step > 0 and self.config['model'] != 'BilinearDiag':
                        sess.run(self.init_inverse_relations)
                        print('Inverse relations are initialized')
                    for l in [1, 2, 3]:
                        summaries.extend(self.online_evaluation(sess, self.train_scores[l], "training path {}".format(l)))
                        summaries.extend(self.online_evaluation(sess, self.valid_scores[l], "validation path {}".format(l)))

                    self.add_summaries(writer, step, [validation_loss_summary] + summaries)

                # Model checkpoint saving
                if local_step % self.config['skip_steps_checkpoint'] == 0 and self.config['checkpoints'] and local_step > 0 \
                        or local_step == self.config['training_steps']:
                    full_save_path = self.saver.save(sess, checkpoint_path, global_step=self.global_step)
                    print("Model saved under: {}".format(full_save_path))

    '''
    EVALUATION
    '''

    def evaluation(self, datapoint):
        e1, e2, e1_domain, e2_domain, path = datapoint

        score_correct = self.predict(e1, path, e2, batches=False)
        scores_e1_corrupted = self.predict(e1_domain, path, e2, batches=False)
        scores_e2_corrupted = self.predict(e1, path, e2_domain, batches=False)

        scores = tf.concat([tf.expand_dims(score_correct, 0), scores_e1_corrupted, scores_e2_corrupted], 0)
        metrics = self.ranking_metrics(scores, order='DESCENDING')

        return metrics

    def online_evaluation(self, sess, evaluation_scores, description):
        batch_metrics = []
        while True:
            try:
                metrics = sess.run(evaluation_scores)
                batch_metrics.append(metrics)

            except tf.errors.OutOfRangeError:
                break

        mean_hits_at_k, mean_quantile, mean_rank = np.nanmean(batch_metrics, axis=0)
        summaries = self.ranking_summaries(description, mean_hits_at_k, mean_quantile, mean_rank)

        return summaries

    def validate_loss(self, sess):
        total_loss = []
        while True:
            try:
                total_loss.append(sess.run(self.validation_loss))
            except tf.errors.OutOfRangeError:
                break
        validation_loss = np.mean(total_loss)

        return self.custom_summary("validation loss", validation_loss)

    def evaluate(self):
        file_path = self.config['path_test_file_neg_domain'][self.args.path_length]
        print('evaluation file:', file_path)
        if self.config['type_matching']:
            generator = Data.evaluation_data_reader(file_path)
        else:
            generator = lambda: self.untyped_evaluation_data(file_path)

        evaluation_data = tf.data.Dataset.from_generator(
            generator, (tf.int32,) * 5, (tf.TensorShape([]),) * 2 + (tf.TensorShape([None]),) * 3).prefetch(1)
        iterator = evaluation_data.make_initializable_iterator()
        datapoint = iterator.get_next()
        evaluation_scores = self.evaluation(datapoint)

        baseline_variables = [self.global_step, self.entity_embedding, self.relation_embedding, self.global_relation_linear_bias]
        evaluation_saver = tf.train.Saver(save_relative_paths=True, var_list=baseline_variables)

        config = tf.ConfigProto(inter_op_parallelism_threads=12, intra_op_parallelism_threads=12)
        with tf.Session(config=config) as sess:
            sess.run(iterator.initializer)
            if self.args.checkpoint == '':
                checkpoint_dir = self.config['checkpoint_dir'] + '/' + self.config['model'] + '/'
                path_prefix = 'path_' if self.args.paths else ''
                description = 'c{}_{}'.format(self.config['id'], self.args.d) if self.args.d else self.describe()
                print(checkpoint_dir + path_prefix + description)
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + path_prefix + description)
            else:
                latest_checkpoint = self.args.checkpoint
            print("Restore model from {}".format(latest_checkpoint))
            evaluation_saver.restore(sess, latest_checkpoint)
            step = sess.run(self.global_step)

            batch_metrics = []

            while True:
                try:
                    metrics = sess.run(evaluation_scores)
                    batch_metrics.append(metrics)

                except tf.errors.OutOfRangeError:
                    break

        mean_hits_at_k, mean_quantile, mean_rank = np.nanmean(batch_metrics, axis=0)

        print('Evaluation of {} with path length {} at step {}'.format(
            latest_checkpoint, self.args.path_length, step))
        print('negative samples: {}'.format('type-matching' if self.config['type_matching'] else 'untyped'))
        print('hits at k: {:0.4f}'.format(mean_hits_at_k))
        print('mean quantile: {:0.4f}'.format(mean_quantile))
        print('mean rank: {:0.4f}'.format(mean_rank))