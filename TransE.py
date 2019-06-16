import tensorflow as tf
from kbc_model import KBC_model

'''
This class implements TransE.
It demonstrates how the CPM can be instantiated with 
KBC models having the following properties:
- predict() supports correctness estimates of paths and has the range [0, 1]
- reduce_path() computes a representation for a given path (e.g. as composition)
'''

class TransE(KBC_model):

    def __init__(self, config, args):
        super().__init__(config, args)
        self.embeddings()

    def embeddings(self):
        with tf.name_scope('embeddings'):
            self.embedding_init = tf.random_normal_initializer(stddev=0.316)

            self.entity_embedding = tf.get_variable('entity_embedding',
                                                    shape=[self.index.entity_count, self.config['embedding_size']],
                                                    initializer=self.embedding_init)

            self.relation_embedding = tf.get_variable('relation_embedding',
                                                      shape=[2*self.index.relation_count, self.config['embedding_size']],
                                                      initializer=self.embedding_init)

            self.global_relation_linear_bias = tf.get_variable('global_relation_linear_bias',
                                                               shape=self.config['embedding_size'],
                                                               initializer=self.embedding_init)

            # This is only needed to predict paths comprising inverse relations after single edge training.
            # Alternatively, inverse triples could be added to the training set.
            self.init_inverse_relations = tf.scatter_update(
                self.relation_embedding,
                tf.range(self.index.relation_count, 2*self.index.relation_count),
                -1 * self.relation_embedding[0:self.index.relation_count])


    def predict(self, e1, path, e2, batches=True, embeddings_given=False, sigmoid=True):
        with tf.name_scope('prediction'):
            if not embeddings_given:
                e1_embedding = tf.nn.embedding_lookup(self.entity_embedding, e1, name='entity_embedding')
                e2_embedding = tf.nn.embedding_lookup(self.entity_embedding, e2, name='entity_embedding')
            else:
                e1_embedding = e1
                e2_embedding = e2
            path_reduced = self.reduce_path(path, batches)
            if self.config['distance_norm'] == 1:
                distance = tf.norm(e1_embedding + path_reduced - e2_embedding, axis=-1, ord=1)
            else:  # distance_norm == 2
                # squared distances seem to be numerically more stable than the L2 norm
                distance = tf.reduce_sum(tf.square(e1_embedding + path_reduced - e2_embedding), axis=-1)
                # eps = 1e-12  # add a small epsilon to avoid NaNs in the gradient computation
                # distance = tf.norm(e1_embedding + path_reduced - e2_embedding + eps, axis=-1, ord='euclidean')
            bias_linear_combination = tf.tensordot(path_reduced, self.global_relation_linear_bias, 1)

            inverse_distance_shifted = bias_linear_combination - distance
            score = tf.sigmoid(inverse_distance_shifted) if sigmoid else inverse_distance_shifted

        return score

    def reduce_path(self, path, batches=True):
        path_axis = 1 if batches else 0
        path_embeddings = tf.nn.embedding_lookup(self.relation_embedding, path, name='relation_embedding')
        path_reduced = tf.reduce_sum(path_embeddings, axis=path_axis)

        return path_reduced
