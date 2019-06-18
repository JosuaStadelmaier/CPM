import os
import itertools

'''
This file specifies all major hyperparameters of the CPM.
Hyperparameters can be given as list to try out all possible combinations.
'''

configurations = {
    'Params_for_BlackboxNLP': {
        'data_set_name': 'FB15K',
        # LEARNING CORRECTNESS SCORES
        'model': ['TransE'],
        'neg_samples': 10,  # number of negative samples per correct path in each batch
        # number of sampled training paths
        # actually not used because path training is restricted to context paths, which are naturally limited in they number
        'path_samples': 1000000,
        # number of epochs to generate training data for (each epoch with different negative samples), applies only to triples
        'neg_samples_epochs': 10,
        # exclude negative samples from valid/test when they are already used in training, applies only to triples
        'evaluation_filter': False,  # cannot be combined with parallel execution
        'type_matching': [True],  # use type-matching corruption or random corruption to generate negative samples
        'path_lengths': range(1, 4),  # sometimes hardcoded, therefore more adaption is needed when changing this
        'optimizer': ['Adam'],
        'normalize': [False],  # letting entity vectors unconstrained improves performance on validation set
        'distance_norm': [2],
        'embedding_size': [100],
        'batch_size': [300],
        'learning_rate': [0.001],
        'path_learning_rate': [0.0001],
        'skip_steps': 20000,  # loss and ranking validation
        'skip_steps_checkpoint': 20000,
        'hits_at_k': 10,
        'shuffle_buffer_size': 10000,
        'training_steps': 80000,
        # LEARNING RELEVANCE SCORES
        # only consider context paths that occur at least x times for other facts
        'context_path_min_frequency': 1,
        'max_context_path_length': 3,
        'context_training_steps': 30000,
        'context_optimizer': ['Adam'],
        'context_learning_rate': [0.001],
        'context_batch_size': 30,
        'context_skip_steps': 500,  # print training loss
        'context_steps_checkpoint': {0, 5000, 10000, 15000, 20000, 25000, 30000},  # loss and ranking validation
        'path_relevance_model': ['DotProduct'],  # see cpm.py for other options
        'context_relevance_regularization': [0.0],
        'fact_regularization': [None],
        # number of single edges to be dropped in each batch, paths are already dropped out in the preprocess_data.py
        'context_dropout_single_edges': [15],  # 50% dropout
        'context_shuffle_buffer_size': 10,  # better shuffle train.txt before generating context paths for each epoch
        'checkpoints': True,
        'summaries': True,
        'print': {'normalize',
                  'learning_rate',
                  'path_learning_rate',
                  'batch_size',
                  'type_matching',
                  'embedding_size'
                  'approx_path_relevance',
                  'context_relevance_regularization',
                  'context_learning_rate',
                  'context_batch_size',
                  'context_dropout_single_edges'}
    }
}

def config_generator(name):
    keys = configurations[name].keys()
    values = [value if type(value) is list else [value] for value in configurations[name].values()]
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        file_paths = get_file_paths(config)
        yield {**config, **file_paths}


def get_file_paths(config):
    data_set_name = config['data_set_name']
    base_dir = 'data/' + data_set_name
    baseline_dir = 'data/' + data_set_name + '/plain_kbc'
    cpm_dir = 'data/' + data_set_name + '/cpm'
    typing = '' if config['type_matching'] else '_untyped'
    return {
        'summary_dir':
            file_path('training_summaries'),
        'checkpoint_dir':
            file_path('checkpoints'),

        # original files
        'original_training_file':
            file_path(base_dir, 'original', 'train.txt'),
        'original_validation_file':
            file_path(base_dir, 'original', 'valid.txt'),
        'original_test_file':
            file_path(base_dir, 'original', 'test.txt'),
        'original_database_file':  # can be downloaded from https://developers.google.com/freebase/
            file_path(base_dir, 'original', 'freebase.gz'),

        # index
        'entity_index_file':
            file_path(base_dir, 'index', 'entity_index.txt'),
        'relation_index_file':
            file_path(base_dir, 'index', 'relation_index.txt'),
        'real_name_index_file':
            file_path(base_dir, 'index', 'real_name_index'),

        # CPM specific indices
        'path_relation_count_file':
            file_path(base_dir, 'index', 'path_relation_count'),
        'path_relation_index_file':
            file_path(base_dir, 'index', 'path_relation_index'),

        # indexed files
        'training_file':
            file_path(baseline_dir, 'train.txt'),
        'validation_file':
            file_path(baseline_dir, 'valid.txt'),
        'test_file':
            file_path(baseline_dir, 'test.txt'),

        # single-edge files for computing the validation loss
        'valid_file_neg_samples':
            file_path(baseline_dir, 'valid_loss{}.txt'.format(typing)),

        # files with paths
        'path_training_file':
            {l: file_path(baseline_dir, 'train_path_{}{}.txt'.format(l, typing)) for l in config['path_lengths']},
        'path_train_file_neg_domain':
            {l: file_path(baseline_dir, 'path_{}_train_ranking{}.txt'.format(l, typing)) for l in config['path_lengths']},
        'path_valid_file_neg_domain':
            {l: file_path(baseline_dir, 'path_{}_valid_ranking{}.txt'.format(l, typing)) for l in config['path_lengths']},
        'path_test_file_neg_domain':
            {l: file_path(baseline_dir, 'path_{}_test_ranking{}.txt'.format(l, typing)) for l in config['path_lengths']},

        # files for learning and evaluating the relevance of paths (type-matching)
        'context_training_file':
            file_path(cpm_dir, 'context_training'),
        'context_train_loss_file':
            file_path(cpm_dir, 'context_training_loss'),
        'context_valid_loss_file':
            file_path(cpm_dir, 'context_valid_loss'),
        'context_train_file_neg_domain':
            file_path(cpm_dir, 'context_ranking_train'),
        'context_valid_file_neg_domain':
            file_path(cpm_dir, 'context_ranking_valid'),
        'context_test_file_neg_domain':
            file_path(cpm_dir, 'context_ranking_test'),

        # files for manually evaluating the relevance of paths
        'context_explanation_samples_file':
            file_path(cpm_dir, 'context_explanation_samples'),
        'explanation_annotations':
            file_path(cpm_dir, 'test_explanation_annotations.csv'),
    }


def file_path(*args):
    current_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(current_dir, *args))
