import tensorflow as tf
import numpy as np
from TransE import TransE
from data_processing import DataProcessing as Data
from operator import itemgetter
import pickle
import subprocess
import sys
import multiprocessing as mp
import os
import gzip
import csv

'''
This class implements the Context Path Model.
Structure:
- CPM specific variables
- Initializers for training and evaluation
- Data for training and evaluation
- Definition of the CPM
- Training
- Evaluation
- Generation of predictions and explanations for annotation study
- Listing the most relevant relation sequences for each relation
- Helper functions
'''


class CPM(TransE):

    def __init__(self, config, args):
        if args.evaluate:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        super().__init__(config, args)

        self.sess_config = tf.ConfigProto(inter_op_parallelism_threads=20, intra_op_parallelism_threads=20)
        self.context_embeddings()

        self.baseline_variables = [self.global_step, self.entity_embedding, self.relation_embedding,
                                   self.global_relation_linear_bias]

        self.path_types = {
            'paths': {'known': (tf.int32,) * self.config['max_context_path_length'],
                      'unknown': (tf.int32,) * self.config['max_context_path_length']
                      },
            'indicators': {
                'known': (tf.bool,) * self.config['max_context_path_length'],
                'unknown': (tf.bool,) * self.config['max_context_path_length']
            },
            'triple': (tf.int32,) * 3
        }

        if args.explain:
            self.explanation_init()
        elif args.evaluate:
            if args.valid:
                self.evaluation_data_set = self.config['context_valid_file_neg_domain']
            else:
                self.evaluation_data_set = self.config['context_test_file_neg_domain']
            self.evaluation_init()
        elif args.analyse:
            self.relevance_analysis()
        else:
            self.training_init()

        self.saver = tf.train.Saver()

    '''
    CPM specific VARIABLES
    '''
    def context_embeddings(self):
        with tf.name_scope('embeddings'):
            self.path_relation_connection = tf.get_variable('path_relation_connection',
                                                            shape=[self.index.relation_count, self.config['embedding_size']],
                                                            initializer=self.embedding_init)
            self.path_relation_bias = tf.get_variable('path_relation_bias',
                                                      shape=[self.config['embedding_size']],
                                                      initializer=self.embedding_init)

    '''
    INITIALIZERS FOR TRAINING AND EVALUATION
    '''

    def training_init(self):
        self.training_data()

        self.training_loss, self.relevance_regularization, self.min_relevance_sum = self.loss(self.training_data_batch)

        if self.args.joint_training:  # joint training of path correctness and relevance
            trainable_variables = None
        else:  # train relevance scores with fixed path correctness scores
            trainable_variables = [self.path_relation_connection, self.path_relation_bias]

        learning_rate = self.config['context_learning_rate']

        self.optimizer = self.optimization(learning_rate, self.config['context_optimizer'])
        self.optimize = self.optimizer.minimize(self.training_loss, global_step=self.global_step, var_list=trainable_variables)

        self.baseline_model = tf.train.Saver(self.baseline_variables)

    def evaluation_init(self):
        self.evaluation_data()
        self.training_evaluation_loss, _, _ = self.loss([self.training_loss_datapoint])
        self.validation_loss, _, _ = self.loss([self.validation_loss_datapoint])
        self.evaluation = {}
        self.evaluation = self.validation_step(self.evaluation_datapoint)
        self.metrics = self.validation()

    def explanation_init(self):
        # manually selected relations for validating relevance scores
        # self.selected_annotation_relations = [193, 1230, 306, 964, 1272, 552, 1108, 640, 464, 856]

        # manually selected relations for testing relevance scores
        self.selected_annotation_relations = [313, 1340, 1113, 167, 683, 432, 0, 121, 576, 570, 1091, 470,
                                              1311, 1058, 329, 356, 40, 633, 773, 1146, 912, 810, 285, 917]

        # select first/second sample (there are two samples for each relation)
        self.annotation_index = {0, 1}
        self.selected_triple_types = {'fact', 'e1_c', 'e2_c'}

        self.explanation_datapoint = self.explanation_data()
        self.explanations = self.predict_scores_and_relevance(self.explanation_datapoint)

    '''
    DATA FOR TRAINING AND EVALUATION
    '''

    def training_data(self):
        # output: (e1_c_indicator, e2_c_indicator, paths, paths_e1_c, paths_e2_c)
        column_types = (tf.bool, tf.bool) + (self.path_types,) * 3
        generator = Data.context_path_reader(self.config['context_training_file'])
        data = tf.data.Dataset.from_generator(generator, column_types).repeat().prefetch(self.config['context_batch_size'])
        # .apply(tf.data.experimental.shuffle_and_repeat(self.config['context_shuffle_buffer_size']))\
        self.training_data_iterator = data.make_initializable_iterator()

        self.training_data_batch = []
        for i in range(self.config['context_batch_size']):
            with tf.name_scope('context_paths_batch_{}'.format(i)):
                datapoint = self.training_data_iterator.get_next()
                self.training_data_batch.append(datapoint)

    def evaluation_data(self):
        with tf.name_scope('context_evaluation_data'):
            # data set for computing the training loss on single edges
            self.training_loss_datapoint = self.loss_evaluation_data(self.config['context_train_loss_file'])

            # data set for computing the validation loss on single edges
            self.validation_loss_datapoint = self.loss_evaluation_data(self.config['context_valid_loss_file'])

            # data sets for computing context ranking metrics on single edges
            file = self.evaluation_data_set
            if self.args.part is not None:
                file += '_part{}'.format(self.args.part)
            print('Evaluating {}'.format(file))
            self.evaluation_datapoint = self.ranking_evaluation_data(Data.context_path_reader(file))

    def loss_evaluation_data(self, file):
        # output: (e1_c_indicator, e2_c_indicator, paths, paths_e1_c, paths_e2_c)
        column_types = (tf.bool, tf.bool) + (self.path_types,) * 3
        generator = Data.context_path_reader(file)
        evaluation_data = tf.data.Dataset.from_generator(generator, column_types).prefetch(1)
        iterator = evaluation_data.make_one_shot_iterator()
        datapoint = iterator.get_next()
        return datapoint

    def ranking_evaluation_data(self, generator):
        # output: (first/last negative sample?, path)
        column_types = (tf.bool, self.path_types)
        evaluation_data = tf.data.Dataset.from_generator(generator, column_types).prefetch(1)
        iterator = evaluation_data.make_one_shot_iterator()
        datapoint = iterator.get_next()
        return datapoint

    def explanation_data(self):
        # data set for performing predictions with explanations
        # output: (paths)
        generator = Data.context_path_explanation_reader(self.config['context_explanation_samples_file'],
                                                         self.selected_annotation_relations, self.annotation_index)
        evaluation_data = tf.data.Dataset.from_generator(generator, self.path_types).prefetch(1)
        iterator = evaluation_data.make_one_shot_iterator()
        datapoint = iterator.get_next()
        return datapoint

    '''
    DEFINITION OF THE CPM
    '''

    # computes c(t, P_t)
    def predict_context_aware(self, paths, validate=False, dropout=False, explain=False):
        e1, e2, r = paths['triple']
        path_lists = paths['paths']
        length_indicators = paths['indicators']

        correctness_scores = []
        relevance_scores = []
        edge_relevance_scores = []
        e1_embedding = tf.nn.embedding_lookup(self.entity_embedding, e1, name='entity_embedding')
        e2_embedding = tf.nn.embedding_lookup(self.entity_embedding, e2, name='entity_embedding')

        if not dropout:
            # dropout the triple itself such that the CPM is forced to also consider longer paths
            # dropout on paths can be applied in the path generation
            score, relevance_score = self.predict_fixed_path_length(e1_embedding, e2_embedding, r,
                tf.expand_dims(tf.expand_dims(r, 0), 0), known_paths='unknown')
            correctness_scores.append(score)
            relevance_scores.append(relevance_score)
            edge_relevance_scores.append(relevance_score)

        for path_type in ['known', 'unknown']:
            for l in range(self.config['max_context_path_length']):
                score, relevance_score = tf.cond(length_indicators[path_type][l],
                    lambda: self.predict_fixed_path_length(e1_embedding, e2_embedding, r, path_lists[path_type][l],
                                                           known_paths=path_type),
                    lambda: (tf.constant([0.0]), tf.constant([0.0])))
                correctness_scores.append(score)
                relevance_scores.append(relevance_score)
                if l == 0:
                    edge_relevance_scores.append(relevance_score)

        scores = tf.concat(correctness_scores, 0)
        scores = tf.check_numerics(scores, 'scores check')
        scores = tf.stop_gradient(scores)
        relevance = tf.concat(relevance_scores, 0)
        edge_relevance = tf.concat(edge_relevance_scores, 0)
        relevance = tf.check_numerics(relevance, 'relevance check')


        if validate and self.args.context_only:
            # context-only setting:
            # only use paths other than t itself if there are such paths with non-zero relevance
            context_available = tf.greater(tf.reduce_sum(relevance[1:]), 0.0)
            relevance = tf.cond(context_available, lambda: tf.concat([tf.constant([0.0]), relevance[1:]], 0),
                                                lambda: tf.concat([tf.constant([1.0]), relevance[1:]], 0))

        else:
            context_available = tf.constant(True)

        # Combine path correctness and path relevance scores to c(t,P_t)
        eps = 1e-10
        normalization = tf.reduce_sum(relevance) + eps
        context_score = tf.tensordot(relevance, scores, 1) / normalization
        context_score = tf.check_numerics(context_score, 'combined check')

        if not validate:
            relevance_regularization = tf.reduce_mean(edge_relevance)
        else:
            relevance_regularization = tf.constant(0.0)

        if validate:
            return context_score, context_available
        elif explain:
            path_lists_explain = [tf.expand_dims(tf.expand_dims(r, 0), 0)]
            for path_type in ['known', 'unknown']:
                for l in range(self.config['max_context_path_length']):
                    # if there exist no paths for a certain length, add a dummy path -1 with relevance 0
                    path_lists_explain.append(
                        tf.cond(length_indicators[path_type][l], lambda: path_lists[path_type][l], lambda: tf.constant([[-1]])))

            _, triple_relevance = self.predict_fixed_path_length(e1_embedding, e2_embedding, r,
                                                                 tf.expand_dims(tf.expand_dims(r, 0), 0), known_paths='unknown')

            return context_score, relevance, scores, path_lists_explain, e1, e2, r, context_available, triple_relevance
        else:
            return context_score, relevance, relevance_regularization, r

    def predict_fixed_path_length(self, e1_embedding, e2_embedding, r, paths_l, known_paths, batches=True):
        if known_paths == 'unknown':
            # predict the correctness of paths if they do not exist in the knowledge graph
            score = self.predict(e1_embedding, paths_l, e2_embedding, batches=batches, embeddings_given=True)
        else:
            # directly assign the correctness score 1.0 to paths that exist in the knowledge graph
            score = tf.fill([tf.shape(paths_l)[0]], 1.0)
        relevance_score = self.path_relevance(r, paths_l, batches=batches)
        return score, relevance_score

    def predict_scores_and_relevance(self, paths):
        context_score, relevance, scores, path_lists, e1, e2, r, context_available, triple_relevance = \
            self.predict_context_aware(paths, explain=True)
        combined = scores * relevance
        triple = (e1, r, e2)

        return [triple, context_score, combined, scores, relevance, path_lists, context_available, triple_relevance]

    def path_relevance(self, r, p, batches=True):
        # several models for learning the relevance of p for t, i.e. \rho(t,p)
        # by representing triples and paths by the relations they consist of
        # more advanced models e.g. for capturing sufficient and necessary conditions could be implemented

        p_embedding = self.reduce_path(p, batches=batches)
        r_embedding = tf.nn.embedding_lookup(self.relation_embedding, r, name='relation_embedding')
        c_embedding = tf.nn.embedding_lookup(self.path_relation_connection, r, name='path_relation_connection')

        # this is the model we use in our paper
        if 'DotProduct' in self.config['path_relevance_model']:
            score = tf.tensordot(p_embedding, c_embedding, 1)

        elif 'BilinearDiag' in self.config['path_relevance_model']:
            score = tf.reduce_sum(r_embedding * c_embedding * p_embedding, axis=-1)

        elif 'TransE' in self.config['path_relevance_model']:
            score = - tf.norm(r_embedding + c_embedding - p_embedding + 1e-8, axis=-1, ord='euclidean')
            # score = -tf.reduce_sum(tf.square(r_embedding + c_embedding - p_embedding), axis=-1)

        elif 'SimpleDiffNorm' in self.config['path_relevance_model']:
            score = - tf.norm(c_embedding - p_embedding + 1e-8, axis=-1, ord='euclidean')

        elif 'SimpleDiffSquared' in self.config['path_relevance_model']:
            score = - tf.reduce_sum(tf.square(c_embedding - p_embedding), axis=-1)

        elif 'SimpleDiff' in self.config['path_relevance_model']:
            score = - tf.reduce_sum(c_embedding - p_embedding, axis=-1)

        elif 'LinCombSquaredDistance' in self.config['path_relevance_model']:
            score = - tf.reduce_sum(c_embedding * tf.square(r_embedding - p_embedding), axis=-1)

        elif 'LinCombAbsDistance' in self.config['path_relevance_model']:
            score = - tf.reduce_sum(c_embedding * tf.abs(r_embedding - p_embedding), axis=-1)

        elif 'LinCombDiff' in self.config['path_relevance_model']:
            score = - tf.reduce_sum(c_embedding * (r_embedding - p_embedding), axis=-1)

        else:  # self.config['path_relevance_model'] == 'Bilinear':  # c_embedding should now be a matrix
            print('Bilinear')
            score_1 = tf.tensordot(p_embedding, c_embedding, 1)
            score_1 = tf.check_numerics(score_1, 'score 1 check')
            score = tf.tensordot(score_1, r_embedding, 1)
            score = tf.check_numerics(score, 'score 2 check')

        if 'no_bias' in self.config['path_relevance_model']:
            relevance_score = tf.exp(score)
        else:
            bias = tf.tensordot(r_embedding, self.path_relation_bias, 1)
            bias = tf.check_numerics(bias, 'bias check')
            relevance_score = tf.exp(bias + score)

        relevance_score = tf.check_numerics(relevance_score, 'relevance check 1')
        return relevance_score

    def loss(self, data_batch):
        loss_batch = []

        def get_loss(paths, correct_path=False, dropout=False):
            context_score, relevance, relevance_norm, r = self.predict_context_aware(paths, dropout=dropout)

            # cross entropy loss
            if correct_path:
                prediction_loss = context_score
                loss = -tf.log(tf.nn.relu(prediction_loss) + 1e-8)
                loss = tf.check_numerics(loss, 'prediction loss correct')
            else:
                prediction_loss = 1 - context_score
                loss = -tf.log(tf.nn.relu(prediction_loss) + 1e-8)
                loss = tf.check_numerics(loss, 'prediction loss wrong')

            # regularization
            if not self.args.evaluate:
                # apply L1 regularization to relevance scores of edges to encourage sparsity
                # and that longer paths are considered
                if self.config['context_relevance_regularization'] > 0.0:
                    loss += self.config['context_relevance_regularization'] * relevance_norm

                # for numerical stability, relevance scores should not become too small or too large
                mean_relevance = tf.reduce_mean(relevance)
                loss += 1000.0 * tf.exp(-1000.0 * mean_relevance)
                loss += 1e-5 * mean_relevance

            return loss, relevance_norm, tf.reduce_sum(relevance)

        def get_combined_loss(paths_e1_c, paths_e2_c, drop):
            loss_e1, relevance_norm_e1, relevance_sum_e1 = get_loss(paths_e1_c, dropout=drop)
            loss_e2, relevance_norm_e2, relevance_sum_e2 = get_loss(paths_e2_c, dropout=drop)
            combined_loss = (loss_e1+loss_e2)/2.0
            combined_relevance_norm = (relevance_norm_e1+relevance_norm_e2)/2.0
            min_relevance_sum = tf.minimum(relevance_sum_e1, relevance_sum_e2)  # only for monitoring
            return combined_loss, combined_relevance_norm, min_relevance_sum

        relevance_regularization_batch = []
        relevance_batch = []
        batch_number = 0
        for batch_element in data_batch:
            with tf.name_scope('loss_context_batch'):

                batch_number += 1
                drop = True if self.config['context_dropout_single_edges'] is not None \
                               and batch_number <= self.config['context_dropout_single_edges'] \
                               and not self.args.evaluate else False

                e1_corrupted, e2_corrupted, paths, paths_e1_c, paths_e2_c = batch_element
                correct_loss, correct_relevance_norm, correct_relevance_sum = \
                    get_loss(paths, correct_path=True, dropout=drop)

                both_corrupted = tf.logical_and(e1_corrupted, e2_corrupted)
                corrupted_loss, corrupted_relevance_norm, corrupted_relevance_sum = tf.case(
                    [(both_corrupted, lambda: get_combined_loss(paths_e1_c, paths_e2_c, drop=drop)),
                        (e1_corrupted, lambda: get_loss(paths_e1_c, dropout=drop)),
                        (e2_corrupted, lambda: get_loss(paths_e2_c, dropout=drop))])

                loss_batch.append(correct_loss+corrupted_loss)
                relevance_regularization_batch.append(correct_relevance_norm + corrupted_relevance_norm)
                relevance_batch.append(tf.minimum(correct_relevance_sum, corrupted_relevance_sum))

        loss = tf.reduce_mean(tf.stack(loss_batch))
        relevance_regularization = tf.reduce_mean(tf.stack(relevance_regularization_batch))
        relevance_min = tf.reduce_min(tf.stack(relevance_batch))

        return loss, relevance_regularization, relevance_min

    '''
    TRAINING
    '''

    def optimization(self, learning_rate, algorithm):
        optimizers = {'Adagrad': tf.train.AdagradOptimizer,
                      'Adadelta': tf.train.AdadeltaOptimizer,
                      'Adam': tf.train.AdamOptimizer}

        optimizer = optimizers[algorithm](learning_rate)

        return optimizer

    def train(self):
        print('Joint training: {}'.format(self.args.joint_training))
        description = 'c{}_{}'.format(self.config['id'], self.args.d) if self.args.d else super().describe()
        context_description = self.args.cd + '_' + str(self.config['id']) if self.args.cd else self.describe()
        path_prefix = 'path_' if self.args.paths else ''
        checkpoint_dir = self.config['checkpoint_dir'] + '/' + self.config['model'] + '/' + path_prefix
        joint_training = '_joint' if 'joint' not in self.args.cd and self.args.joint_training else ''
        checkpoint_path = checkpoint_dir + context_description + joint_training + '/model-at-step'
        with tf.Session(config=self.sess_config) as sess:
            # Data initialization
            sess.run(self.training_data_iterator.initializer)

            # Restore model
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + context_description + joint_training)
            if latest_checkpoint is not None:
                print("Restore CPM from {}".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
            else:
                # Optimizer parameter initialization
                sess.run(tf.variables_initializer(self.optimizer.variables()))

                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + description)
                print("Restore plain KBC model from {}".format(latest_checkpoint))
                self.baseline_model.restore(sess, latest_checkpoint)

                sess.run(self.path_relation_connection.initializer)
                sess.run(self.path_relation_bias.initializer)

            print('Start training')
            # Training
            total_loss = 0.0
            total_fact_reg = 0.0
            min_relevance = 1e20
            for local_step in range(self.config['context_training_steps'] + 1):
                step = sess.run(self.global_step)

                # Optimization
                loss_batch, relevance_reg, min_relevance_new, _ = sess.run(
                    [self.training_loss, self.relevance_regularization, self.min_relevance_sum, self.optimize])

                # Training loss reports
                total_loss += loss_batch
                total_fact_reg += relevance_reg
                min_relevance = min_relevance if min_relevance < min_relevance_new else min_relevance_new
                if step % self.config['context_skip_steps'] == 0 and local_step > 0 \
                        or step == self.config['context_training_steps']:
                    print("\nstep: {}".format(step))

                    training_loss = total_loss / self.config['context_skip_steps']
                    fact_regularization = total_fact_reg / self.config['context_skip_steps']
                    print("{}: {:0.4f}".format('training loss', training_loss))
                    print("{}: {}".format('fact regularization', fact_regularization))
                    print("{}: {}".format('min relevance', min_relevance))
                    total_loss = 0.0
                    total_fact_reg = 0.0
                    min_relevance = 1e20

                # Checkpoint saving and evaluation reports
                if self.config['checkpoints'] and (local_step in self.config['context_steps_checkpoint']):
                    full_save_path = self.saver.save(sess, checkpoint_path, global_step=self.global_step)
                    print("Model saved under: {}".format(full_save_path))

                    joint_training = ['--joint_training'] if '--joint_training' not in sys.argv and self.args.joint_training else []

                    cmd = ['python'] + sys.argv + ['--valid', '--evaluate', '-i', str(self.config['id'])] + joint_training
                    if '&' in cmd:
                        cmd.remove("&")
                    subprocess.Popen(cmd)

    '''
    EVALUATION
    '''

    def validation(self):
        with tf.name_scope('validation'):
            self.scores = tf.placeholder(tf.float32, shape=[None])
            metrics = tf.cond(tf.size(self.scores) > 1,
                              lambda: self.ranking_metrics(self.scores, order='DESCENDING'),
                              lambda: [tf.to_float(float('NaN'))]*3)
        return metrics

    def validation_step(self, datapoint):
        with tf.name_scope('context_validation'):
            separator, paths = datapoint
            score, context_available = self.predict_context_aware(paths, validate=True)
        return [score, separator, context_available]

    def validate(self, sess):
        summaries = []
        batch_metrics = []
        evaluated_facts = 0
        skipped_facts = 0
        neg_samples_skipped = []
        step = sess.run(self.global_step)
        # for count in range(1500):
        dataset_description = 'valid' if self.args.valid else 'test'
        count = 0
        while True:
            count += 1
            if count % 100 == 0:
                print('{} ranking evaluation on {} triples ({} skipped) at step {}'.format(
                    dataset_description, count, skipped_facts, step))
                mean_hits_at_k, mean_quantile, mean_rank = np.nanmean(batch_metrics, axis=0)
                print('HITS@10: {:0.4f},  MQ: {:0.4f},  MR: {:0.4f}'.format(mean_hits_at_k, mean_quantile, mean_rank))
            skip_fact = False
            evaluated_neg_samples = 0
            skipped_neg_samples = 0
            fact = True
            try:
                last_neg_sample = False
                scores = []
                while not last_neg_sample:
                    score, last_neg_sample, context_available = sess.run(self.evaluation)
                    if fact and not context_available:
                        skip_fact = True

                    if not fact:
                        if not context_available:
                            skipped_neg_samples += 1
                        else:
                            evaluated_neg_samples += 1

                    scores.append(score)

                    fact = False

                metrics = sess.run(self.metrics, feed_dict={self.scores: scores})
                batch_metrics.append(metrics)
                neg_samples_skipped.append(skipped_neg_samples/(evaluated_neg_samples+skipped_neg_samples))
                if not skip_fact:
                    evaluated_facts += 1
                else:
                    skipped_facts += 1
            except tf.errors.OutOfRangeError:
                break

        print('Average of skipped neg samples: {}'.format(sum(neg_samples_skipped) / len(neg_samples_skipped)))
        print('Facts without context: {}'.format(skipped_facts))
        print('Facts with context: {}'.format(evaluated_facts))

        mean_hits_at_k, mean_quantile, mean_rank = np.nanmean(batch_metrics, axis=0)

        print('\nRanking evaluation at step {}'.format(step))
        summary = self.ranking_summaries(dataset_description, mean_hits_at_k, mean_quantile, mean_rank)
        summaries += summary

        return summaries

    def evaluate_loss(self, sess, loss, description):
        total_loss = []
        while True:
            try:
                total_loss.append(sess.run(loss))
            except tf.errors.OutOfRangeError:
                break

        validation_loss = np.mean(total_loss)

        return self.custom_summary(description, validation_loss)

    def ranking_evaluation_process(self, summary_queue):
        with tf.Session(config=self.sess_config) as sess:
            self.restore_context_model(sess)

            # Evaluation reports
            ranking_summaries = self.validate(sess)
            for summary in ranking_summaries:
                summary_queue.put(summary)

    def evaluate_online(self):
        summary_description = self.args.d + '/' if self.args.cd else ''
        writer = tf.summary.FileWriter(self.config['summary_dir'] + '/' + summary_description + self.describe())

        # ranking evaluation
        summary_queue = mp.Queue()
        evaluation_process = mp.Process(target=self.ranking_evaluation_process, args=(summary_queue,))
        evaluation_process.start()

        # loss evaluation
        with tf.Session(config=self.sess_config) as sess:
            self.restore_context_model(sess)
            step = sess.run(self.global_step)
            training_loss_summary = self.evaluate_loss(sess, self.training_evaluation_loss, 'training loss')
            validation_loss_summary = self.evaluate_loss(sess, self.validation_loss, 'validation loss')
            summaries = [training_loss_summary, validation_loss_summary]

        evaluation_process.join()

        while not summary_queue.empty():
            summaries.append(summary_queue.get(block=False))

        self.add_summaries(writer, step, summaries)

    '''
    GENERATION OF PREDICTIONS AND EXPLANATIONS FOR ANNOTATION STUDY
    '''

    def predict_and_explain(self):
        with tf.Session(config=self.sess_config) as sess:
            # some data structures for calculating annotation stats
            self.explanation_lengths = {1: 0, 2: 0, 3: 0}
            self.displayed_relevance = []

            self.annotations = {}
            self.missing_annotations = []
            self.annotation_count = 0
            self.no_context_filter = 0

            self.global_annotation_stats = {'e': [], 'ae': [], 'n': [], 'an': [], 's': [], 'as': [], 'nr': []}
            self.global_annotation_relevance = {'e': [], 'ae': [], 'n': [], 'an': [], 's': [], 'as': [], 'nr': []}

            self.global_annotation_stats_pro = {'e': 0, 'ae': 0, 'n': 0, 'an': 0, 's': 0, 'as': 0, 'nr': 0}
            self.global_annotation_stats_con = self.global_annotation_stats_pro.copy()

            self.global_annotation_count = {'': 0, 'e': 0, 'ae': 0, 'n': 0, 'an': 0, 's': 0, 'as': 0, 'nr': 0, 'u': 0}
            self.global_annotation_count_pro = self.global_annotation_count.copy()
            self.global_annotation_count_con = self.global_annotation_count.copy()

            self.pro_fractions = 0
            self.con_fractions = 0

            # mapping from fine to coarse annotation scheme
            self.annotation_mapping = {
                '': 'todo',
                'e': 'equivalent',
                'ae': 'probable',
                'n': 'probable',
                'an': 'probable',
                's': 'probable',
                'as': 'probable',
                'nr': 'unrelated',
                'u': 'ignore'
            }

            self.skipped_relations = []

            skipped = 0
            count = 0

            # Data initialization
            self.restore_context_model(sess)

            # Load annotations
            with open(self.config['explanation_annotations'], 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    annotation = row[0]
                    relation = int(row[1])
                    relation_sequence = tuple(np.array(row[2:]).astype(np.int32))
                    self.annotations.setdefault(relation, {})
                    if relation_sequence not in self.annotations[relation]:
                        self.annotations[relation][relation_sequence] = annotation

            # Perform predictions and generate explanations
            if self.args.verbose:
                print('Predictions and explanations of the CPM')
                print('See our paper "Modeling Paths for Explainable Knowledge Base Completion" for details.\n')
                print('Entities are printed in \033[1mbold\033[0m.')
                print('For corrupted paths, the possible points of corruption are printed in \x1B[3mitalic\033[0m.')
                print('Inverse relations are marked with a minus.')

            total_counter = 0
            with gzip.open(self.config['context_explanation_samples_file'], 'rb') as file:
                while True:
                    try:
                        explanation_paths = pickle.load(file)
                        total_counter += 1
                        if len(self.selected_annotation_relations) > 0 \
                                and explanation_paths['relation'] not in self.selected_annotation_relations \
                                or explanation_paths['index'] not in self.annotation_index:
                            continue
                        explanation_available = self.explain(sess, explanation_paths, self.args.verbose)
                        if explanation_available:
                            count += 1
                        else:
                            skipped += 1
                    except EOFError:
                        break

            # Calculate stats for the annotated explanations
            print("Evaluation of CPM explanations:")
            print('Fine-grained annotation statistics for pro and con arguments')

            if self.pro_fractions > 0:
                print('\npro analysis:')
                print('class: \tavg relevance fraction / # pairs')
                for annotation in self.global_annotation_stats_pro:
                    normalized_relevance_fraction = self.global_annotation_stats_pro[annotation] / self.pro_fractions
                    print('{}: \t{:0.2f} / {}'.format(self.annotation_output(annotation),
                                                    normalized_relevance_fraction,
                                                    self.global_annotation_count_pro[annotation]))

            if self.con_fractions > 0:
                print('\ncon analysis:')
                print('class: \tavg relevance fraction / # pairs')
                for annotation in self.global_annotation_stats_con:
                    normalized_relevance_fraction = self.global_annotation_stats_con[annotation] / self.con_fractions
                    print('{}: \t{:0.2f} / {}'.format(self.annotation_output(annotation),
                                                    normalized_relevance_fraction,
                                                    self.global_annotation_count_con[annotation]))

            if self.pro_fractions > 0 or self.con_fractions > 0:
                relevance_coarse = {'equivalent': [], 'probable': [], 'unrelated': []}
                mean_fraction_coarse = {'equivalent': 0.0, 'probable': 0.0, 'unrelated': 0.0}
                annotation_coarse = {'equivalent': 0, 'probable': 0, 'unrelated': 0}

                print('\nAnnotation statistics with the annotation classes as described in the paper:')
                for annotation in self.global_annotation_stats:
                    mean_fraction = sum(self.global_annotation_stats[annotation])/len(self.global_annotation_stats[annotation])
                    coarse_annotation_class = self.annotation_mapping[annotation]
                    relevance_coarse[coarse_annotation_class] += self.global_annotation_relevance[annotation]
                    mean_fraction_coarse[coarse_annotation_class] += mean_fraction
                    annotation_coarse[coarse_annotation_class] += self.global_annotation_count[annotation]

                print('class: \t\tavg relevance / avg relevance fraction / share of pairs / # pairs')
                for annotation in relevance_coarse:
                    if len(relevance_coarse[annotation]) > 0:
                        print('{}:\t{:0.2f} / {:0.2f} / {:0.2f} / {}'.format(
                            annotation,
                            sum(relevance_coarse[annotation])/len(relevance_coarse[annotation]),
                            mean_fraction_coarse[annotation],
                            annotation_coarse[annotation]/sum(annotation_coarse.values()),
                            annotation_coarse[annotation]))
                    else:
                        print('{}:\t-'.format(annotation))

            # general stats on the annotation
            print('\nmissing annotations: {}'.format(len(self.missing_annotations )))
            print('used annotations: {}'.format(self.annotation_count))

            print('\nnumber of displayed relations: {}'.format(count))
            print('number of skipped relations: {}'.format(skipped))
            print('number of skipped because of missing context: {}'.format(self.no_context_filter))
            for length in self.explanation_lengths:
                print('paths of length {} per triple / in total: {:0.2f} / {}'.format(
                    length, self.explanation_lengths[length]/count, self.explanation_lengths[length]))
            print('average relevance displayed: {:0.2f}'.format(sum(self.displayed_relevance)/len(self.displayed_relevance)))

            print('\nskipped triples ({}):'.format(len(self.skipped_relations)))
            print([(self.index.resolve_names(triple[0]), triple[1]) for triple in self.skipped_relations])

            # add missing annotations to the annotation file s.t. they can be completed
            if self.args.annotate:
                with open(self.config['explanation_annotations'], 'a') as f:
                    writer = csv.writer(f)
                    for row in self.missing_annotations:
                        writer.writerow(row)

    # explanation for one triple
    def explain(self, sess, explanation_data, output=True):
        explanations = explanation_data['explanation_paths']
        corrupted = explanation_data['description']

        triple, context_score, combined, scores, relevance, path_indices, context_available, triple_relevance = \
            sess.run(self.explanations)
        e1, r, e2 = triple

        if corrupted not in self.selected_triple_types:
            return False

        # exclude triples for which no context paths exist
        if not context_available:
            self.no_context_filter += 1
            return False

        # exclude triples for which only low relevant context paths exist
        if triple_relevance[0] / 9.0 > np.sum(relevance[1:]):
            self.no_context_filter += 1
            self.skipped_relations.append((triple, corrupted))
            return False

        combined = combined[1:]
        scores = scores[1:]
        relevance = relevance[1:]

        path_indices = path_indices[1:]

        # normalize relevance scores
        normalized_relevance = relevance/np.sum(relevance)

        # flatten paths of different lengths into one list
        paths = []
        for path_l in path_indices:
            for path in path_l:
                paths.append(path.tolist())
        result_list_transposed = list(map(list, zip(combined, scores, normalized_relevance, paths, relevance)))

        results_filtered = [x for x in result_list_transposed if x[2] > 0.05]
        # sort by combined DSC
        pro_results = [x for x in results_filtered if x[1] > 0.5]
        pro_results = sorted(pro_results, key=itemgetter(0), reverse=True)
        # first sort by combined ASC and then by relevance DSC
        con_results = [x for x in results_filtered if x[1] < 0.5]
        con_results = sorted(con_results, key=itemgetter(2), reverse=True)

        if len(pro_results) < 1 and len(con_results) < 1:
            return False

        if output:
            if corrupted == 'fact':
                description = 'correct fact'
                print('-' * 100 + '\n')
            elif corrupted == 'e1_c':
                description = 'fact with e1 corrupted'
                print('-' * 100 + '\n')
            else:  # corrupted == 2
                description = 'fact with e2 corrupted'
                print('-' * 100 + '\n')

            triple_names = self.index.resolve_names(triple)
            # print(r)
            print('{}:\n{}'.format(description, triple_names))
            print('score: {:0.2f}'.format(context_score))

        displayed_relevance = 0

        annotation_stats = {'': 0, 'e': 0, 'ae': 0, 'n': 0, 'an': 0, 's': 0, 'as': 0, 'nr': 0, 'u': 0}

        annotation_stats_pro = annotation_stats.copy()
        annotation_stats_con = annotation_stats.copy()

        if len(pro_results) >= 1:
            if output:
                print('\nPRO:')
            for result in pro_results[:20]:
                path_relations = tuple(result[3])
                displayed_relevance += result[2]
                annotation = self.annotations[r][path_relations] \
                    if r in self.annotations and path_relations in self.annotations[r] else ''
                if (r not in self.annotations or r in self.annotations and path_relations not in self.annotations[r]) \
                        and (r,) != path_relations:
                    annotation_row = ['', r] + list(path_relations)
                    if annotation_row not in self.missing_annotations:
                        self.missing_annotations.append(annotation_row)

                if (r,) != path_relations and annotation not in {'', 'u'}:
                    self.explanation_lengths[len(path_relations)] += 1
                    annotation_stats_pro[annotation] += result[2]
                    annotation_stats[annotation] += result[2]
                    self.global_annotation_count_pro[annotation] += 1
                    self.global_annotation_count[annotation] += 1
                    self.annotation_count += 1
                    self.global_annotation_relevance[annotation] += [result[2]]

                if output and (r,) != path_relations and (not self.args.filter_annotated or annotation == '') and annotation != 'u':
                    fine_grained_class = " (more specific: {})".format(self.annotation_output(annotation)) \
                        if annotation not in {'e', 'nr', ''} else ''
                    print('\nscore: {:0.2f}   relevance: {:0.2f}   annotation: {}{}'.format(
                        result[1], result[2], self.annotation_mapping[annotation], fine_grained_class))
                    print(Data.get_explanation_output(path_relations, explanations, e1, e2, r, self.index))

        if len(con_results) >= 1:
            if output:
                print('\nCON:')
            for result in con_results[:20]:
                path_relations = tuple(result[3])
                displayed_relevance += result[2]
                annotation = self.annotations[r][path_relations] \
                    if r in self.annotations and path_relations in self.annotations[r] else ''
                if (r not in self.annotations or r in self.annotations and path_relations not in self.annotations[r]) \
                        and (r,) != path_relations:
                    annotation_row = ['', r] + list(path_relations)
                    if annotation_row not in self.missing_annotations:
                        self.missing_annotations.append(annotation_row)

                if (r,) != path_relations and annotation not in {'', 'u'}:
                    self.explanation_lengths[len(path_relations)] += 1
                    annotation_stats_con[annotation] += result[2]
                    annotation_stats[annotation] += result[2]
                    self.global_annotation_count_con[annotation] += 1
                    self.global_annotation_count[annotation] += 1
                    self.annotation_count += 1
                    self.global_annotation_relevance[annotation] += [result[2]]

                if output and (r,) != path_relations and (not self.args.filter_annotated or annotation == '') and annotation != 'u':
                    fine_grained_class = " (more specific: {})".format(self.annotation_output(annotation)) \
                        if annotation not in {'e', 'nr', ''} else ''
                    print('\nscore: {:0.2f}   relevance: {:0.2f}   annotation: {}{}'.format(
                        result[1], result[2], self.annotation_mapping[annotation], fine_grained_class))
                    print(Data.get_explanation_output(path_relations, explanations, e1, e2, r, self.index))

        pro_relevance = sum(annotation_stats_pro.values())
        con_relevance = sum(annotation_stats_con.values())
        combined_relevance = pro_relevance + con_relevance

        if pro_relevance > 0:
            pro_fraction = pro_relevance / combined_relevance
            self.pro_fractions += pro_fraction
            for annotation in self.global_annotation_stats_pro:
                annotation_fraction = annotation_stats_pro[annotation]/pro_relevance
                weighted_annotation_fraction = pro_fraction * annotation_fraction
                self.global_annotation_stats_pro[annotation] += weighted_annotation_fraction

        if con_relevance > 0:
            con_fraction = con_relevance / combined_relevance
            self.con_fractions += con_fraction
            for annotation in self.global_annotation_stats_con:
                annotation_fraction = annotation_stats_con[annotation]/con_relevance
                weighted_annotation_fraction = con_fraction * annotation_fraction
                self.global_annotation_stats_con[annotation] += weighted_annotation_fraction

        if pro_relevance > 0 or con_relevance > 0:
            for annotation in self.global_annotation_stats:
                annotation_fraction = annotation_stats[annotation] / combined_relevance
                self.global_annotation_stats[annotation] += [annotation_fraction]

        if len(pro_results) >= 1 or len(con_results) >= 1:
            self.displayed_relevance.append(displayed_relevance)

        return True

    def annotation_output(self, annotation):
        annotation_meaning = {
            '': 'todo',
            'e': 'equivalent',
            'ae': 'almost equivalent',
            'n': 'necessary',
            'an': 'almost necessary',
            's': 'sufficient',
            'as': 'almost sufficient',
            'nr': 'unrelated',
            'u': 'unclear'
        }
        return annotation_meaning[annotation]

    '''
    LISTING THE MOST RELEVANT RELATION SEQUENCES FOR EACH RELATION
    '''

    def relevance_analysis(self):
        self.analysis_paths = tf.placeholder(tf.int32)
        self.analysis_relation = tf.placeholder(tf.int32)
        self.analysis_relevance_score = self.path_relevance(self.analysis_relation, self.analysis_paths)

    def analyse_relevance_scores(self):
        print('Analyse relevance scores')
        with tf.Session(config=self.sess_config) as sess:
            self.restore_context_model(sess)

            with open(self.config['path_relation_index_file'], 'rb') as file:
                path_relation_index = pickle.load(file)
                relation_relevance_scores = {}

                for r in path_relation_index.keys():
                # for r in [313, 1305, 1342, 1311, 1308, 1306, 1168, 1146, 1063, 734]:
                    for l in range(1, self.config['max_context_path_length'] + 1):
                        paths = list(path_relation_index[r][l])
                        if len(paths) > 0:
                            relevance = sess.run(self.analysis_relevance_score,
                                                 feed_dict={self.analysis_paths: paths, self.analysis_relation: r})
                            for i in range(len(paths)):
                                relation_relevance_scores.setdefault(r, []).append([paths[i], relevance[i]])

                count = 0
                skipped = 0
                for relation, paths_with_relevance in relation_relevance_scores.items():
                    sorted_by_relevance = sorted(paths_with_relevance, key=itemgetter(1), reverse=True)
                    path_output = []
                    highest_relevance = sorted_by_relevance[0][1]
                    for path in sorted_by_relevance[:20]:
                        path_relevance = path[1]
                        if path_relevance > highest_relevance/5.0:
                            path_output.append('relevance: {}\n{}\n'.format(
                                path_relevance, self.index.resolve_relation_path_names(path[0])))
                    if len(path_output) > 1:
                        count += 1
                        print('\n' + '-'*10 + '\n')
                        print('relation: {} - {}\n'.format(relation, self.index.get_relation_name(relation)))
                        for line in path_output:
                            print(line)
                    else:
                        skipped += 1

                print('number of displayed relations: {}'.format(count))
                print('number of skipped relations with one relevant path: {}'.format(skipped))

    '''
    HELPER FUNCTIONS
    '''

    def restore_context_model(self, sess):
        context_description = self.args.cd + '_' + str(self.config['id']) if self.args.cd else self.describe()
        path_prefix = 'path_' if self.args.paths else ''
        joint_training = '_joint' if 'joint' not in self.args.cd and self.args.joint_training else ''
        checkpoint_dir = self.config['checkpoint_dir'] + '/' + self.config['model'] + '/' + path_prefix
        if self.args.checkpoint == '':
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir + context_description + joint_training)
        else:
            latest_checkpoint = checkpoint_dir + context_description + joint_training + '/model-at-step-' + self.args.checkpoint
        print("Restore CPM {}".format(latest_checkpoint))
        self.saver.restore(sess, latest_checkpoint)

    def describe(self):
        return '{}{}_dropout={}_relevance-model={}_c-l-rate={}_e-dim={}_c-b-size={}_relevance_reg={}_{}'.format(
            self.args.cd + '_' if self.args.cd else self.args.d + '_',
            self.config['id'],
            self.config['context_dropout_single_edges'],
            self.config['path_relevance_model'],
            self.config['context_learning_rate'],
            self.config['embedding_size'],
            self.config['context_batch_size'],
            self.config['context_relevance_regularization'],
            '_path' if self.args.paths else ''
        )
