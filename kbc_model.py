import tensorflow as tf
from index import Index
import numpy as np

'''
This class provides general KBC helper methods for calculating ranking metrics
and saving training summaries.
The CPM can be instantiated with various KBC models by extending this class,
see TransE.py for an example.
'''

class KBC_model(object):

    def __init__(self, config, args):
        tf.reset_default_graph()
        self.config = config
        self.args = args
        self.index = Index.load(self.config)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def ranking_metrics(self, scores, order):
        with tf.name_scope('ranking_metrics'):
            ranking = tf.contrib.framework.argsort(scores, direction=order)
            # first element in scores is the correct entity
            # tf.where(tf.equal(ranking, 0)[0, 0] returns the index of the correct entity in ranking
            rank_of_correct_entity = tf.cast(tf.where(tf.equal(ranking, 0))[0, 0], tf.int32)
            # highest rank corresponds to 0 => use hits at k-1
            hit_at_k = tf.cond(rank_of_correct_entity < self.config['hits_at_k'], lambda: 1.0, lambda: 0.0)
            quantile = (tf.size(scores) - rank_of_correct_entity - 1) / (tf.size(scores) - 1)
        return [tf.to_float(hit_at_k), tf.to_float(quantile), tf.to_float(rank_of_correct_entity)+1.0]

    # the same metrics implemented using numpy, mainly for performance comparison and verification
    def ranking_metrics_np(self, scores, order='ascending'):
        if len(scores) > 1:
            sorted_scores = np.sort(scores)
            # first element in scores is the correct entity
            rank = np.argwhere(sorted_scores == scores[0])[0][0] + 1
            if order == 'descending':
                rank = len(scores)-rank+1
            quantile = (len(scores)-rank)/(len(scores)-1)
            hit = 1.0 if rank <= 10 else 0.0
            return [hit, quantile, rank]
        else:
            return [np.nan]*3

    def ranking_summaries(self, description, mean_hits_at_k, mean_quantile, mean_rank):
        summaries = [
            self.custom_summary('{} hits at {}'.format(description, self.config['hits_at_k']), mean_hits_at_k),
            self.custom_summary(description + ' mean quantile', mean_quantile),
            self.custom_summary(description + ' mean rank', mean_rank)
        ]
        return summaries

    def custom_summary(self, name, value):
        print("{}: {:0.4f}".format(name, value))
        return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

    def add_summaries(self, writer, step, summaries):
        if self.config['summaries']:
            for summary in summaries:
                writer.add_summary(summary, global_step=step)
        writer.flush()

    def describe(self):
        return 'c{}_type={}_l-rate={}_e-dim={}_b-size={}_norm={}'.format(
            self.config['id'],
            self.config['type_matching'],
            self.config['learning_rate'],
            self.config['embedding_size'],
            self.config['batch_size'],
            self.config['normalize'],
        )
