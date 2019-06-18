from config import config_generator
import csv
import numpy as np
import random
from index import Index
import argparse
from collections import deque
import pickle
import gzip

'''
This class handles all data processing for the Context Path Model
and the plain KBC models.
Structure:
- Indexing
- Knowledge graph generation
- Plain KBC training data generation
- Plain KBC evaluation data generation
- Plain KBC path traversing
- Context path indexing
- Context path generation
- Explanation data generation
- File reader and writer
- Real name index for Freebase MIDs
- Debugging
'''

class DataProcessing(object):

    def __init__(self, config):
        self.config = config

        if args.index:
            # indexing of the knowledge base
            print('Create index')
            self.index = Index.create(config)
            # use Index.load(config) instead if index already exists but indexed files need to be recreated
            print('Index data sets')
            training_set = self.index_data_set(config['original_training_file'])
            self.write_data_set(training_set, config['training_file'])
            _, entities, _ = self.get_relation_domains([config['training_file']])
            validation_set = self.index_data_set(config['original_validation_file'], entities)
            self.write_data_set(validation_set, config['validation_file'])
            test_set = self.index_data_set(config['original_test_file'], entities)
            self.write_data_set(test_set, config['test_file'])
        else:
            # get index and relation domains for knowledge base
            self.index = Index.load(config)
            self.relation_domains, self.entities, self.relations = self.get_relation_domains(
                [config['training_file'], config['validation_file'], config['test_file']])

            # build knowledge graphs representing the knowledge base
            if not args.update_cpm_index and not args.path_relation_stats and not args.real_name_index:
                print('Build knowledge graphs')
                self.train_knowledge_graph = self.build_knowledge_graph([config['training_file']])
                self.full_knowledge_graph = self.build_knowledge_graph([config['training_file'],
                                                                        config['validation_file'],
                                                                        config['test_file']])
                self.knowledge_graph_for_context_paths = {}
                # this is needed for filtering out negative samples from validation/test that are used in single edge training
                self.corrupted_training_knowledge_graph = {}

        # Create single edge data sets for the underlying KBC model for c(t), e.g. TransE
        if args.plain:

            # Training data sets
            print('\nPrepare single edge training data set')
            single_edge_training_data_set = self.create_data_set(config['training_file'], False, 1, train=True)
            self.write_data_set(single_edge_training_data_set, config['path_training_file'][1])

            # Loss evaluation data sets
            print('\nGet triples for validation loss data set')
            validation_loss_set = self.create_data_set(config['validation_file'], config['evaluation_filter'], 1)
            self.write_data_set(validation_loss_set, config['valid_file_neg_samples'])

            # Ranking evaluation data sets
            print('\nSample triples for training evaluation data set')
            triples = self.parallel_data_reader(config['training_file'], relative_sample_size=0.05)
            training_evaluation_set = self.triple_evaluation_set(triples)
            self.write_data_set(training_evaluation_set, config['path_train_file_neg_domain'][1])

            print('\nSample triples for validation data set')
            triples = self.parallel_data_reader(config['validation_file'], relative_sample_size=None)
            validation_set = self.triple_evaluation_set(triples, evaluation_filter=config['evaluation_filter'])
            self.write_data_set(validation_set, config['path_valid_file_neg_domain'][1])

            print('\nGet triples for test data set')
            triples = self.parallel_data_reader(config['test_file'])
            test_set = self.triple_evaluation_set(triples, evaluation_filter=config['evaluation_filter'])
            self.write_data_set(test_set, config['path_test_file_neg_domain'][1])

        # Path generation for training and evaluating c(p)
        if args.paths:

            # Rather use more paths than sample more negatives per paths (in contrast to edge training)
            self.config['neg_samples_epochs'] = 1

            # Train path generation
            self.knowledge_graph_for_context_paths = self.train_knowledge_graph

            # length 2
            print('\nGenerate context paths of length 2')
            context_paths = self.context_path_generator(config['training_file'], relative_sample_size=None,
                                                                path_length=2, train=True, max_paths=30)
            path_data_set = self.create_data_set(None, False, 2, context_paths=context_paths)
            self.write_data_set(path_data_set, self.indexed_file(config['path_training_file'][2]))

            # length 3
            print('\nGenerate context paths of length 3')
            context_paths = self.context_path_generator(config['training_file'], relative_sample_size=None,
                                                                path_length=3, train=True, max_paths=10)
            path_data_set = self.create_data_set(None, False, 3, context_paths=context_paths)
            self.write_data_set(path_data_set, self.indexed_file(config['path_training_file'][3]))

            # Train evaluation path generation

            # length 2
            print('\nPrepare context paths for train eval data with paths of length 2')
            context_paths = self.context_path_generator(config['training_file'], relative_sample_size=0.06,
                                                                     path_length=2, train=True, max_paths=1)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_train_file_neg_domain'][2]))

            # length 3
            print('\nPrepare context paths for train eval data with paths of length 3')
            context_paths = self.context_path_generator(config['training_file'], relative_sample_size=0.04,
                                                        path_length=3, train=True, max_paths=1)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_train_file_neg_domain'][3]))

            # Valid path generation
            self.knowledge_graph_for_context_paths = self.build_knowledge_graph([config['training_file'],
                                                                                 config['validation_file']])

            # length 2
            print('\nPrepare valid data set with paths of length 2')
            context_paths = self.context_path_generator(config['validation_file'], relative_sample_size=None,
                                                        path_length=2, max_paths=8)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_valid_file_neg_domain'][2]))

            # length 3
            print('\nPrepare valid data set with paths of length 3')
            context_paths = self.context_path_generator(config['validation_file'], relative_sample_size=None,
                                                        path_length=3, max_paths=3)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_valid_file_neg_domain'][3]))

            # Test path generation
            self.knowledge_graph_for_context_paths = self.build_knowledge_graph([config['training_file'],
                                                                                 config['test_file']])

            # length 2
            print('\nPrepare test data set with paths of length 2')
            context_paths = self.context_path_generator(config['test_file'], relative_sample_size=None,
                                                        path_length=2, max_paths=50)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_test_file_neg_domain'][2]))

            # length 3
            print('\nPrepare test data set for paths of length 3')
            context_paths = self.context_path_generator(config['test_file'], relative_sample_size=None,
                                                        path_length=3, max_paths=10)
            path_evaluation_set = self.path_evaluation_set(context_paths)
            self.write_data_set(path_evaluation_set, self.indexed_file(config['path_test_file_neg_domain'][3]))

        # create (filtered) index of context path-relation combinations
        # 1) get frequencies of context path-relation combinations
        # 2) use frequencies to filter out context paths that only occur very rarely for a given relation
        elif args.cpm_index:
            print('\nCount occurrences of path-relation combinations')
            self.knowledge_graph_for_context_paths = self.train_knowledge_graph
            self.get_path_relation_count()
            print('\nCreate index of filtered path-relation combinations')
            self.generate_path_relation_index()

        # update (filtered) index of context path-relation combinations after changing context_path_min_frequency
        elif args.update_cpm_index:
            print('Create index of filtered path-relation combinations')
            self.generate_path_relation_index()

        # print stats on the combinations of relations and context paths
        elif args.path_relation_stats:
            self.path_relation_stats(config['path_relation_index_file'])

        # data sets for training and evaluating the full CPM (i.e. the relevance part with trained path correctness)
        elif args.cpm:
            self.path_relation_index = self.load_path_relation_index()
            self.knowledge_graph_for_context_paths = self.train_knowledge_graph
            if not config['type_matching']:
                print('Only type-matching corruption is supported by the CPM.\n'
                      'The explanation for non type-matching negative samples is just the type-mismatch itself.')

            print('\nPrepare relevance training data')
            self.context_path_writer(config['training_file'],
                                     self.indexed_file(config['context_training_file']),
                                     relative_sample_size=None, sample=True)

            print('\nPrepare relevance training evaluation loss data')
            self.context_path_writer(config['training_file'],
                                     self.indexed_file(config['context_train_loss_file']),
                                     relative_sample_size=0.1, sample=False)

            print('\nPrepare relevance validation loss data')
            self.context_path_writer(config['validation_file'],
                                     self.indexed_file(config['context_valid_loss_file']),
                                     relative_sample_size=None, sample=False)

            print('\nPrepare relevance training evaluation data')
            context_data = list(self.parallel_data_reader(config['training_file'], relative_sample_size=0.003))
            self.context_evaluation_writer(self.single_edge_generator(data=context_data),
                                           self.indexed_file(config['context_train_file_neg_domain']),
                                           relative_neg_sample_size=0.1)

            print('\nPrepare relevance validation data')
            context_data = list(self.parallel_data_reader(config['validation_file'], relative_sample_size=0.05))
            self.context_evaluation_writer(self.single_edge_generator(data=context_data),
                                           self.indexed_file(config['context_valid_file_neg_domain']),
                                           relative_neg_sample_size=0.5)

            print('\nPrepare relevance test data')
            context_data = list(self.parallel_data_reader(config['test_file'], relative_sample_size=0.01))
            self.context_evaluation_writer(self.single_edge_generator(data=context_data),
                                           self.indexed_file(config['context_test_file_neg_domain']))

        elif args.explanations:
            self.path_relation_index = self.load_path_relation_index()
            self.knowledge_graph_for_context_paths = self.train_knowledge_graph

            print('\nPrepare explanation data')
            self.explanation_data_writer(config['test_file'], config['context_explanation_samples_file'], relative_sample_size=0.01)

        # extract real names for entities from the full FB15K data set,
        # which can be downloaded from https://developers.google.com/freebase/
        elif args.real_name_index:
            self.generate_real_name_index()

    '''
    INDEXING
    '''

    def index_data_set(self, file_path, entities=None):
        data_set = []
        with open(file_path, 'r') as file:
            csv_file = csv.reader(file, delimiter='\t')
            for line in csv_file:
                indexed_triple = self.index.apply(line)
                data_set.append(indexed_triple)
                # exclude technical relations because they are not that interesting for explanations
                # if self.config['data_set_name'] != 'FB15K':
                #     data_set.append(indexed_triple)
                # elif ('/common/' not in line[1] and '/dataworld/' not in line[1]) \
                #         and (entities is None or ((indexed_triple[0] in entities) and (indexed_triple[2] in entities))):
                #     data_set.append(indexed_triple)
        return data_set

    # relation domains
    def get_relation_domains(self, data_files):
        relation_domains = {}
        entities = set()
        relations = set()
        for file in data_files:
            for e1, r, e2 in self.data_reader(file):
                entities.update([e1, e2])
                relations.add(r)
                if r in relation_domains:
                    relation_domains[r]['left'].add(e1)
                    relation_domains[r]['right'].add(e2)
                else:
                    relation_domains[r] = {'left': {e1}, 'right': {e2}}

        return relation_domains, entities, relations

    '''
    KNOWLEDGE GRAPH GENERATION
    '''

    def build_knowledge_graph(self, data_files):
        knowledge_graph = {}
        for data_file in data_files:
            for triple in self.data_reader(data_file):
                self.update_knowledge_graph(knowledge_graph, triple)
        return knowledge_graph

    def update_knowledge_graph(self, knowledge_graph, triple):
        e1, r, e2 = triple
        if e1 in knowledge_graph:
            if r in knowledge_graph[e1]['out']:
                knowledge_graph[e1]['out'][r].add(e2)
            else:
                knowledge_graph[e1]['out'][r] = {e2}
        else:
            knowledge_graph[e1] = {'out': {r: {e2}}, 'in': {}}

        if e2 in knowledge_graph:
            if r in knowledge_graph[e2]['in']:
                knowledge_graph[e2]['in'][r].add(e1)
            else:
                knowledge_graph[e2]['in'][r] = {e1}
        else:
            knowledge_graph[e2] = {'out': {}, 'in': {r: {e1}}}

    '''
    PLAIN KBC TRAINING DATA GENERATION
    '''

    # loss data set creation
    def create_data_set(self, file, evaluation_filter, path_length, context_paths=None, train=False):
        if self.config['type_matching']:
            return self.create_data_set_type_matching(file, evaluation_filter, path_length, context_paths, train)
        else:
            return self.create_data_set_untyped(file, evaluation_filter, path_length, context_paths, train)

    # loss data set creation (type-matching corruption)
    def create_data_set_type_matching(self, file, evaluation_filter, path_length, context_paths=None, train=False):
        data_set = []
        if context_paths is None:
            source = self.single_edge_generator(file_path=file, evaluation_filter=evaluation_filter) if path_length == 1 \
                else self.path_generator(path_length)
        else:
            source = self.paths_with_type_matching_neg_domains(context_paths)
        i = 0
        for e1, e2, e1_c_domain, e2_c_domain, path in source:
            e1_c = self.sample_corrupted_entities(e1_c_domain) if len(e1_c_domain) > 0 else None
            e2_c = self.sample_corrupted_entities(e2_c_domain) if len(e2_c_domain) > 0 else None
            data_set.append({'e1': e1, 'e2': e2, 'e1_c': e1_c, 'e2_c': e2_c, 'path': path})
            i += 1
            if i % 10000 == 0: print(i)

        type_match_1 = 0
        type_match_2 = 0
        data_set_expanded = []
        neg_samples = self.config['neg_samples']
        for i in range(self.config['neg_samples_epochs']):
            for line in data_set:
                e1, e2, e1_c, e2_c, path = line['e1'], line['e2'], line['e1_c'], line['e2_c'], line['path']
                if e1_c and e2_c:
                    data_set_expanded.append([2, e1, e2] + e1_c[i*neg_samples:(i+1)*neg_samples]
                                                         + e2_c[i*neg_samples:(i+1)*neg_samples] + path)
                elif e1_c:
                    type_match_1 += 1
                    data_set_expanded.append([0, e1, e2] + e1_c[i*neg_samples:(i+1)*neg_samples] + [0]*neg_samples + path)
                elif e2_c:
                    type_match_2 += 1
                    data_set_expanded.append([1, e1, e2] + [0]*neg_samples + e2_c[i*neg_samples:(i+1)*neg_samples] + path)

        if self.config['type_matching']:
            print('type match trivial e1: {}'. format(type_match_1))
            print('type match trivial e2: {}'. format(type_match_2))

        # add negative samples used in training to self.corrupted_training_knowledge_graph,
        # which can be used to exclude such negatives from validation/test data
        if self.config['evaluation_filter'] and path_length == 1 and train:
            for line in data_set:
                e1, e2, e1_c, e2_c, path = line['e1'], line['e2'], line['e1_c'], line['e2_c'], line['path']
                for i in range(self.config['neg_samples_epochs'] * neg_samples):
                    if e1_c:
                        self.update_knowledge_graph(self.corrupted_training_knowledge_graph, (e1_c[i], path[0], e2))
                    if e2_c:
                        self.update_knowledge_graph(self.corrupted_training_knowledge_graph, (e1, path[0], e2_c[i]))

        return data_set_expanded

    # loss data set creation (random corruption)
    def create_data_set_untyped(self, file, evaluation_filter, path_length, context_paths=None, train=False):
        data_set = []
        if context_paths is None:
            if path_length > 1:
                # self.path_generator(path_length) needs to be adapted to untyped negative samples, not needed for CPM
                raise NotImplementedError
            source = self.single_edge_generator_untyped(file_path=file, evaluation_filter=evaluation_filter)
        else:
            source = self.paths_with_untyped_domains(context_paths)
        i = 0

        for e1, e2, e1_domain, e2_domain, path in source:
            e1_c_domain = self.entities - e1_domain
            e2_c_domain = self.entities - e2_domain

            e1_c = self.sample_corrupted_entities(e1_c_domain)
            e2_c = self.sample_corrupted_entities(e2_c_domain)
            data_set.append({'e1': e1, 'e2': e2, 'e1_c': e1_c, 'e2_c': e2_c, 'path': path})
            i += 1
            if i % 10000 == 0: print(i)

        data_set_expanded = []
        neg_samples = self.config['neg_samples']
        for i in range(self.config['neg_samples_epochs']):
            for line in data_set:
                e1, e2, e1_c, e2_c, path = line['e1'], line['e2'], line['e1_c'], line['e2_c'], line['path']
                data_set_expanded.append([2, e1, e2]
                                         + e1_c[i*neg_samples:(i+1)*neg_samples]
                                         + e2_c[i*neg_samples:(i+1)*neg_samples]
                                         + path)

        # add negative samples used in training to self.corrupted_training_knowledge_graph,
        # which can be used to exclude such negatives from validation/test data
        if self.config['evaluation_filter'] and path_length == 1 and train:
            for line in data_set:
                e1, e2, e1_c, e2_c, path = line['e1'], line['e2'], line['e1_c'], line['e2_c'], line['path']
                for i in range(self.config['neg_samples_epochs'] * neg_samples):
                    self.update_knowledge_graph(self.corrupted_training_knowledge_graph, (e1_c[i], path[0], e2))
                    self.update_knowledge_graph(self.corrupted_training_knowledge_graph, (e1, path[0], e2_c[i]))

        return data_set_expanded

    '''
    PLAIN KBC EVALUATION DATA GENERATION
    '''

    def triple_evaluation_set(self, generator, evaluation_filter=None):
        if self.config['type_matching']:
            return self.evaluation_set(self.single_edge_generator(data=generator, evaluation_filter=evaluation_filter))
        else:
            return self.evaluation_set(self.single_edge_generator_untyped(data=generator, evaluation_filter=evaluation_filter))

    def path_evaluation_set(self, generator):
        if self.config['type_matching']:
            return self.evaluation_set(self.paths_with_type_matching_neg_domains(generator))
        else:
            return self.evaluation_set(self.paths_with_untyped_domains(generator))

    def evaluation_set(self, generator):
        domain_sizes_1 = []
        domain_sizes_2 = []
        under_10_samples = 0
        type_match_trivial_count = 0
        i = 0
        for e1, e2, e1_domain, e2_domain, path in generator:

            if len(e1_domain) == 0 and len(e2_domain) == 0:
                type_match_trivial_count += 1
            else:
                e1_domain = list(e1_domain)
                e2_domain = list(e2_domain)
                domain_sizes_1.append(len(e1_domain))
                domain_sizes_2.append(len(e2_domain))
                if len(e1_domain) + len(e2_domain) <= 9:
                    under_10_samples += 1
                i += 1
                if i % 1000 == 0:
                    print(i)

                yield [e1, e2, len(e1_domain), len(e2_domain)] + e1_domain + e2_domain + path

        print('number of correct paths: {}'.format(i))
        if self.config['type_matching']:
            print('mean domain size for e1: {}'.format(sum(domain_sizes_1)/len(domain_sizes_1)))
            print('mean domain size for e2: {}'.format(sum(domain_sizes_2)/len(domain_sizes_2)))
            print('mean domain size (union): {}'.format(sum(domain_sizes_2+domain_sizes_1)/len(domain_sizes_1)))
            print('triples with 0 < domain size < 10: {} ({:0.1f}%)'.format(under_10_samples, 100*under_10_samples/i))
            print("type-match trivial triples (domain size = 0): {}".format(type_match_trivial_count))

    def paths_with_type_matching_neg_domains(self, paths):
        for e1, p, e2 in paths:
            e1_c_domain, e2_c_domain = self.get_corrupted_path_endings(e1, p, e2)
            yield (e1, e2, e1_c_domain, e2_c_domain, p)

    def paths_with_untyped_domains(self, paths):
        for e1, p, e2 in paths:
            e1_domain = self.get_correct_path_endings(self.full_knowledge_graph, p, e2=e2)
            e2_domain = self.get_correct_path_endings(self.full_knowledge_graph, p, e1=e1)
            yield (e1, e2, e1_domain, e2_domain, p)

    def single_edge_generator(self, file_path='', data=None, evaluation_filter=False, parallelize=None, relative_sample_size=None):
        source = data if data else self.parallel_data_reader(file_path, relative_sample_size, parallelize)
        i = -1
        for e1, r, e2 in source:
            i += 1
            e1_c_domain, e2_c_domain = self.get_corrupted_path_endings(e1, [r], e2)

            if evaluation_filter:  # filter out corrupted values that are used in training
                e1_c_train = self.get_correct_path_endings(self.corrupted_training_knowledge_graph, [r], e2=e2)
                e2_c_train = self.get_correct_path_endings(self.corrupted_training_knowledge_graph, [r], e1=e1)
                e1_c_domain -= e1_c_train
                e2_c_domain -= e2_c_train

            yield e1, e2, e1_c_domain, e2_c_domain, [r]

    def single_edge_generator_untyped(self, file_path='', data=None, evaluation_filter=False, parallelize=None,
                                      relative_sample_size=None):
        source = data if data else self.parallel_data_reader(file_path, relative_sample_size, parallelize)
        i = -1
        for e1, r, e2 in source:
            i += 1

            # get all correct entities for the path endings using the full knowledge graph
            e1_domain = self.get_correct_path_endings(self.full_knowledge_graph, [r], e2=e2)
            e2_domain = self.get_correct_path_endings(self.full_knowledge_graph, [r], e1=e1)

            if evaluation_filter:  # filter out corrupted values that are used in training
                e1_c_train = self.get_correct_path_endings(self.corrupted_training_knowledge_graph, [r], e2=e2)
                e2_c_train = self.get_correct_path_endings(self.corrupted_training_knowledge_graph, [r], e1=e1)
                e1_domain += e1_c_train
                e2_domain += e2_c_train

            yield e1, e2, e1_domain, e2_domain, [r]

    '''
    PLAIN KBC PATH TRAVERSING
    '''

    def path_generator(self, path_length):
        for i in range(self.config['path_samples']):
            yield self.sample_path(path_length)

    # randomly sample a path, not used because we focus on the subset of context paths
    def sample_path(self, path_length):
        path = []
        e1 = random.sample(self.entities, 1)[0]
        e = e1
        for _ in range(path_length):
            direction = random.choice(['out', 'in'])
            if not list(self.train_knowledge_graph[e][direction].keys()):
                direction = 'out' if direction == 'in' else 'in'  # flip direction if there is no suitable edge for it
            r = random.choice(list(self.train_knowledge_graph[e][direction].keys()))
            e = random.choice(list(self.train_knowledge_graph[e][direction][r]))
            # relations and their inverse have two separate representations (with the same initialization)
            if direction == 'in':
                r = self.get_inverse_relation(r)
            path.append(r)
        e2 = e
        e1_c_domain, e2_c_domain = self.get_corrupted_path_endings(e1, path, e2)
        return e1, e2, e1_c_domain, e2_c_domain, path

    # sample corrupted entities from their domain and repeat them if the domain is too small
    def sample_corrupted_entities(self, domain):
        domain = list(domain)
        negative_samples = []
        negative_sample_count = self.config['neg_samples_epochs'] * self.config['neg_samples']
        if negative_sample_count > len(domain):
            negative_samples += domain * (self.config['neg_samples_epochs'] * self.config['neg_samples'] // len(domain))
            negative_sample_count %= len(domain)
        negative_samples += random.sample(domain, negative_sample_count)
        return negative_samples

    def get_corrupted_path_endings(self, e1, path, e2):
        # get all correct entities for the path endings using the full knowledge graph
        correct_start_entities = self.get_correct_path_endings(self.full_knowledge_graph, path, e2=e2)
        correct_end_entities = self.get_correct_path_endings(self.full_knowledge_graph, path, e1=e1)

        # get corrupted entities: subtract the correct entities from the relation domain
        # we want the domain of the left side of path[0] => take the right side if path[0] is an inverse relation
        r_first, r_first_side = self.get_composite_relation_index(path[0], ['right', 'left'])
        # we want the domain of the right side of path[-1] => take the left side if path[-1] is an inverse relation
        r_last, r_last_side = self.get_composite_relation_index(path[-1], ['left', 'right'])
        corrupted_start_entities = self.relation_domains[r_first][r_first_side] - correct_start_entities
        corrupted_end_entities = self.relation_domains[r_last][r_last_side] - correct_end_entities
        return corrupted_start_entities, corrupted_end_entities

    def get_correct_path_endings(self, knowledge_graph, path, e1=None, e2=None):
        found_entities_new = {e1 if e1 is not None else e2}
        reverse = e1 is None
        relations = reversed(path) if reverse else path
        for r in relations:
            r, direction = self.get_composite_relation_index(r, ['in', 'out'])
            if reverse:
                direction = 'in' if direction == 'out' else 'out'
            found_entities_old = found_entities_new
            found_entities_new = set()
            for e in found_entities_old:
                if e in knowledge_graph and r in knowledge_graph[e][direction]:
                    found_entities_new |= knowledge_graph[e][direction][r]
        return found_entities_new

    def get_inverse_relation(self, r):
        if r < self.index.relation_count:
            return r + self.index.relation_count
        else:
            return r - self.index.relation_count

    def get_composite_relation_index(self, r, direction):
        if r >= self.index.relation_count:
            r -= self.index.relation_count
            side = direction[0]
        else:
            side = direction[1]
        return r, side

    '''
    CONTEXT PATH INDEXING
    '''

    def load_path_relation_index(self):
        with open(self.config['path_relation_index_file'], 'rb') as file:
            return pickle.load(file)

    def get_path_relation_count(self):
        j = 0
        path_count = {r: {l: {} for l in range(1, self.config['max_context_path_length'] + 1)} for r in self.relations}
        for e1, r, e2 in self.data_reader(self.config['training_file']):

            # Using context paths of corrupted triples probably makes no sense for the standard CPM.
            # The standard CPM uses incorrect paths only as evidence for incorrect triples.
            # For counting the relation sequences, we only consider correct paths.
            # For a possible extension of the CPM that models negative correlations,
            # we could also consider correct context paths corresponding to incorrect triples here.

            triple = e1, r, e2
            correct_paths = self.get_paths_to_triple(triple, sample=False)

            j += 1
            if j % 10000 == 0:
                print('fact {}'.format(j))
                for l in range(1, self.config['max_context_path_length']+1):
                    number_of_combinations = sum([len(path_count[relation][l]) for relation in self.relations])
                    print('combinations of length {}: {}'.format(l, number_of_combinations))

                with open(self.config['path_relation_count_file'], 'wb') as index_file:
                    pickle.dump(path_count, index_file)

            if correct_paths is None:
                continue

            correct_paths = correct_paths['paths']['known']

            # get path occurrences for each relation
            # the relation itself is not in the path list, but it is also indexed
            path_count[r][1].setdefault((r,), 0)
            path_count[r][1][(r,)] += 1
            for l in range(1, self.config['max_context_path_length']+1):
                for p in correct_paths[l-1]:
                    path_count[r][l].setdefault(p, 0)
                    path_count[r][l][p] += 1

        with open(self.config['path_relation_count_file'], 'wb') as index_file:
            pickle.dump(path_count, index_file)

        for l in range(1, self.config['max_context_path_length'] + 1):
            counts = []
            counts_10 = []
            for r in self.relations:
                if len(path_count[r][l]) == 0:
                    counts.append(len(path_count[r][l]))
                if len(path_count[r][l]) > 0 and len(path_count[r][l]) <= 10:
                    counts_10.append(len(path_count[r][l]))
            print('relations without paths of length {}: {}'.format(l, len(counts)))
            print('relations with 1 to 10 paths of length {}: {}'.format(l, len(counts_10)))

    def generate_path_relation_index(self):
        with open(self.config['path_relation_count_file'], 'rb') as file:
            path_count = pickle.load(file)
            # generate index with a subset of frequent paths
            index = {r: {l: set() for l in range(1, self.config['max_context_path_length'] + 1)} for r in self.relations}
            i = 0
            s = 0
            for r in self.relations:
                for path_length in range(1, self.config['max_context_path_length'] + 1):
                    for p, count in path_count[r][path_length].items():
                        # do not filter the facts itself
                        if count < self.config['context_path_min_frequency'] and path_length != 1:
                            s += 1
                            continue
                        index[r][path_length].add(p)
                        i += 1

            print('number of path-relation connections: {}'.format(i))
            print('number of skipped path-relation connections: {}'.format(s))

            with open(self.config['path_relation_index_file'], 'wb') as index_file:
                pickle.dump(index, index_file)

    def path_relation_stats(self, index_file):
        with open(index_file, 'rb') as file:
            index = pickle.load(file)
            i = 0
            i_10 = 0
            combination_count = 0
            for r in self.relations:
                for l in range(1, self.config['max_context_path_length'] + 1):
                    combination_count += len(index[r][l])
                if 0 == len(index[r][2]):
                    i += 1
                if 0 < len(index[r][2]) <= 10:
                    i_10 += 1

            print('context_path_min_frequency: {}'.format(self.config['context_path_min_frequency']))
            print('total number of combinations: {}'.format(combination_count))
            print('number of relations without paths: {}'.format(i))
            print('number of relations with 1 to 10 paths: {}'.format(i_10))

            for i in [1, 2, 3]:
                combinations = [len(index[relation][i]) for relation in self.relations if len(index[relation][i]) > 0]
                print('average number of paths of length {} per relation: {}'.format(i, sum(combinations)/len(combinations)))

    '''
    CONTEXT PATH GENERATION
    '''

    # generate context paths which can later be used as training/evaluation data for c(p)
    def context_path_generator(self, source_file, relative_sample_size=None, path_length=None, train=False, max_paths=None):
        self.path_relation_index = self.load_path_relation_index()

        i = 0
        self.true_path_count = {1: [], 2: [], 3: []}
        self.false_path_count = {1: [], 2: [], 3: []}

        for e1, r, e2 in self.parallel_data_reader(source_file, relative_sample_size=relative_sample_size):
            triple = e1, r, e2

            i += 1
            if i % 5000 == 0:
                print('generated context paths for {} facts'.format(i))

            paths = self.get_paths_to_triple(triple, sample=False, path_length=path_length, path_generation=True)

            if paths is None:
                continue

            known_paths = {1: paths['paths']['known'][1], 2: paths['paths']['known'][2]}

            # Ensure that config['shuffle_buffer_size'] is large enough when many paths are sampled per triple.

            if path_length is not None:
                if max_paths is None:
                    max_paths = 100 if path_length == 2 else 40
                if path_length == 2:
                    path_list = [random.sample(known_paths[1], max_paths) if len(known_paths[1]) > max_paths else known_paths[1]]
                else:
                    path_list = [random.sample(known_paths[2], max_paths) if len(known_paths[2]) > max_paths else known_paths[2]]

            else:
                context_paths_2 = random.sample(known_paths[1], 100) if len(known_paths[1]) > 100 else known_paths[1]
                context_paths_3 = random.sample(known_paths[2], 10) if len(known_paths[2]) > 10 else known_paths[2]
                path_list = [context_paths_2, context_paths_3]

            for context_paths in path_list:
                for context_path in context_paths:
                    p = list(context_path)
                    # use only paths that do not occur in the training data for the test or validation set
                    if train or e2 not in self.get_correct_path_endings(self.train_knowledge_graph, p, e1=e1):
                        yield (e1, p, e2)

            if i % 1000 == 0 and args.verbose:
                print('Generated context paths for {} facts.'.format(i))
                print('Average number of context paths per fact:')
                for l in [1, 2, 3]:
                    print('correct context paths of length {}: {}'.format(
                        l, sum(self.true_path_count[l]) / len(self.true_path_count[l])))
                    print('potentially incorrect context paths of length {}: {}'.format(
                        l, sum(self.false_path_count[l]) / len(self.false_path_count[l])))

        print('Generated context paths for {} facts'.format(i))

    def context_path_writer(self, source_file, output_file, relative_sample_size=None, sample=True):
        i = 0
        self.true_path_count = {1: [], 2: [], 3: []}
        self.false_path_count = {1: [], 2: [], 3: []}

        with gzip.open(output_file, 'ab') as file:
            empty_paths = {
                'paths': {
                    'known': ([], ) * self.config['max_context_path_length'],
                    'unknown': ([], ) * self.config['max_context_path_length']
                },
                'indicators': {
                    'known': (False,) * self.config['max_context_path_length'],
                    'unknown': (False,) * self.config['max_context_path_length']
                },
                'triple': (0, 0, 0)
            }
            for e1, e2, e1_c_domain, e2_c_domain, path in self.single_edge_generator(
                    source_file, relative_sample_size=relative_sample_size):
                r = path[0]
                triple = e1, r, e2

                paths = self.get_paths_to_triple(triple, sample=True)

                if paths is None:
                    continue

                paths_e1_c = empty_paths
                e1_corrupted = False
                if len(e1_c_domain) > 0:
                    triple_e1_c = random.choice(list(e1_c_domain)), r, e2
                    paths_e1_c_temp = self.get_paths_to_triple(triple_e1_c, sample=True)
                    if paths_e1_c_temp is not None:
                        e1_corrupted = True
                        paths_e1_c = paths_e1_c_temp

                paths_e2_c = empty_paths
                e2_corrupted = False
                if len(e2_c_domain) > 0:
                    triple_e2_c = e1, r, random.choice(list(e2_c_domain))
                    paths_e2_c_temp = self.get_paths_to_triple(triple_e2_c, sample=True)
                    if paths_e2_c_temp is not None:
                        e2_corrupted = True
                        paths_e2_c = paths_e2_c_temp

                if not e1_corrupted and not e2_corrupted:
                    continue

                pickle.dump((e1_corrupted, e2_corrupted, paths, paths_e1_c, paths_e2_c), file)

                i += 1
                if i % 1000 == 0:
                    print('Generated context paths for {} facts.'.format(i))
                    if args.verbose:
                        print('Average number of context paths per fact:')
                        for l in [1, 2, 3]:
                            print('correct context paths of length {}: {}'.format(
                                l, sum(self.true_path_count[l])/len(self.true_path_count[l])))
                            print('potentially incorrect context paths of length {}: {}'.format(
                                l, sum(self.false_path_count[l])/len(self.false_path_count[l])))

            print('Generated context paths for {} facts.'.format(i))

    def context_evaluation_writer(self, generator, output_file, relative_neg_sample_size=None):
        with gzip.open(output_file, 'ab') as file:
            j = 0
            for e1, e2, e1_c_domain, e2_c_domain, path in generator:
                r = path[0]
                triple = e1, r, e2

                paths = self.get_paths_to_triple(triple, sample=False)

                if paths is None:
                    continue

                j += 1
                if j % 2 == 0:
                    print(j)

                if relative_neg_sample_size is not None and len(e1_c_domain)+len(e2_c_domain) > 100:
                    absolute_sample_size_e1 = round(len(e1_c_domain) * relative_neg_sample_size)
                    absolute_sample_size_e2 = round(len(e2_c_domain) * relative_neg_sample_size)
                    e1_c_domain = random.sample(e1_c_domain, absolute_sample_size_e1)
                    e2_c_domain = random.sample(e2_c_domain, absolute_sample_size_e2)

                corrupted_triples = []
                if len(e1_c_domain) > 0:
                    for e1_c in e1_c_domain:
                        corrupted_triples.append((e1_c, r, e2))
                if len(e2_c_domain) > 0:
                    for e2_c in e2_c_domain:
                        corrupted_triples.append((e1, r, e2_c))

                if len(corrupted_triples) > 0:
                    pickle.dump((False, paths), file)
                    for i in range(len(corrupted_triples)):
                        paths_c = self.get_paths_to_triple(corrupted_triples[i], skip=False, sample=False)
                        last_neg_sample = True if i == len(corrupted_triples) - 1 else False
                        pickle.dump((last_neg_sample, paths_c), file)

            print('TOTAL NUMBER: {}'.format(j))

    def get_paths_to_triple(self, triple, sample=True, skip=True, path_length=None, path_generation=False):
        e1, r, e2 = triple

        known_paths = {l: set() for l in range(1, self.config['max_context_path_length'] + 1)}
        unknown_paths = {l: set() for l in range(1, self.config['max_context_path_length'] + 1)}

        # paths of length 1
        single_edges_known = set()
        single_edges_unknown = set()
        if not path_generation:  # unordered known context paths of length 1 are just the edges of the knowledge graph
            for path, reached_entities in self.path_traversal(deque([r]), e1, e1, e2, 1, r, False, single_edge=True):
                if e2 in reached_entities:
                    single_edges_known.add(path)

                if not args.cpm_index and not path_generation:
                    if e2 not in reached_entities and path in self.path_relation_index[r][1]:
                        relation_norm, side = self.get_composite_relation_index(path[0], ['left', 'right'])
                        if e2 in self.relation_domains[relation_norm][side]:
                            single_edges_unknown.add(path)

            if not args.cpm_index and not path_generation:
                for path, reached_entities in self.path_traversal(deque([r]), e2, e1, e2, 1, r, False, single_edge=True):
                    path = (self.get_inverse_relation(path[0]),)

                    if e1 not in reached_entities and path in self.path_relation_index[r][1]:
                        relation_norm, side = self.get_composite_relation_index(path[0], ['right', 'left'])
                        if e1 in self.relation_domains[relation_norm][side]:
                            single_edges_unknown.add(path)

        # paths of length 2
        if not path_generation or path_length == 2:
            left_side = {1: {}}
            left_side_inv = {1: {}}
            for path, reached_entities in self.path_traversal(deque([r]), e1, e1, e2, 1, r, False):
                left_side_inv[len(path)][path] = reached_entities
                for reached_entity in reached_entities:
                    left_side[len(path)].setdefault(reached_entity, set()).add(path)

            right_side = {1: {}}
            right_side_inv = {1: {}}
            for path, reached_entities in self.path_traversal(deque([r]), e2, e1, e2, 1, r, False):
                right_side_inv[len(path)][path] = reached_entities
                for reached_entity in reached_entities:
                    right_side[len(path)].setdefault(reached_entity, set()).add(path)

            known_paths_2 = self.merge_paths(left_side, right_side, 2)
        else:
            known_paths_2 = set()

        # paths of length 3
        if not path_generation or path_length == 3:
            path_len = 3
            left_max_path_length = (path_len + 1) // 2
            left_side = {l: {} for l in range(1, left_max_path_length + 1)}
            left_side_inv = {l: {} for l in range(1, left_max_path_length + 1)}
            for path, reached_entities in self.path_traversal(deque([r]), e1, e1, e2, left_max_path_length, r, sample):
                left_side_inv[len(path)][path] = reached_entities
                for reached_entity in reached_entities:
                    left_side[len(path)].setdefault(reached_entity, set()).add(path)

            right_max_path_length = (path_len + 1) // 2
            right_side = {l: {} for l in range(1, left_max_path_length + 1)}
            right_side_inv = {l: {} for l in range(1, right_max_path_length + 1)}
            for path, reached_entities in self.path_traversal(deque([r]), e2, e1, e2, right_max_path_length, r, sample):
                right_side_inv[len(path)][path] = reached_entities
                for reached_entity in reached_entities:
                    right_side[len(path)].setdefault(reached_entity, set()).add(path)

            known_paths_3 = self.merge_paths(left_side, right_side, path_len)
        else:
            known_paths_3 = set()

        # dropout 50% of context paths if there are more than three of them
        known_paths[1] = random.sample(single_edges_known, len(single_edges_known) // 2)\
            if sample and len(single_edges_known) > 3 else single_edges_known
        known_paths[2] = random.sample(known_paths_2, len(known_paths_2) // 2)\
            if sample and len(known_paths_2) > 3 else known_paths_2
        known_paths[3] = random.sample(known_paths_3, len(known_paths_3) // 2)\
            if sample and len(known_paths_3) > 3 else known_paths_3

        if not args.cpm_index and not path_generation:
            # only consider paths consisting of relation sequences of the path-relation index
            potential_unknown_paths_2 = self.path_relation_index[r][2] - known_paths_2
            potential_unknown_paths_3 = self.path_relation_index[r][3] - known_paths_3

            # union of context paths for which initial or final entities are not empty
            unknown_paths_2 = self.merge_false_paths_l(left_side_inv, right_side_inv, 2, potential_unknown_paths_2) \
                               | self.merge_false_paths_r(left_side_inv, right_side_inv, 2, potential_unknown_paths_2)
            unknown_paths_3 = self.merge_false_paths_l(left_side_inv, right_side_inv, 3, potential_unknown_paths_3) \
                               | self.merge_false_paths_r(left_side_inv, right_side_inv, 3, potential_unknown_paths_3)

            # dropout 50% of context paths if there are more than three of them
            unknown_paths[1] = random.sample(single_edges_unknown, len(single_edges_unknown) // 2)\
                if sample and len(single_edges_unknown) > 3 else single_edges_unknown
            unknown_paths[2] = random.sample(unknown_paths_2, len(unknown_paths_2) // 2)\
                if sample and len(unknown_paths_2) > 3 else unknown_paths_2
            unknown_paths[3] = random.sample(unknown_paths_3, len(unknown_paths_3) // 2)\
                if sample and len(unknown_paths_3) > 3 else unknown_paths_3

        if args.verbose:
            for l in [1, 2, 3]:
                self.true_path_count[l].append(len(known_paths[l]))
                self.false_path_count[l].append(len(unknown_paths[l]))

        known_paths = tuple(list(known_paths[l]) for l in range(1, self.config['max_context_path_length'] + 1))
        unknown_paths = tuple(list(unknown_paths[l]) for l in range(1, self.config['max_context_path_length'] + 1))
        known_path_indicators = [len(paths_l) > 0 for paths_l in known_paths]
        unknown_path_indicators = [len(paths_l) > 0 for paths_l in unknown_paths]

        paths_exist = True in known_path_indicators + unknown_path_indicators

        if not paths_exist and skip:
            return None

        return {'paths':
                    {'known': known_paths, 'unknown': unknown_paths},
                'indicators':
                    {'known': tuple(known_path_indicators), 'unknown': tuple(unknown_path_indicators)},
                'triple': (e1, e2, r)}

    # merge paths found by bidirectional search from e1 and e2 by considering the intersection of reached entities
    def merge_paths(self, left_side, right_side, l):
        left_l = (l + 1) // 2
        right_l = l // 2
        left_to_right_paths = set()
        # right_to_left_paths = set() # these paths are technically slightly different from those in the inverse direction
        intersection = left_side[left_l].keys() & right_side[right_l].keys()
        for entity in intersection:
            left_paths = left_side[left_l][entity]
            right_paths = right_side[right_l][entity]
            for p_left in left_paths:
                for p_right in right_paths:
                    if p_left[-1] != p_right[-1]:  # Avoid paths consisting of a relation followed by its inverse relation
                        left_to_right_paths.add(p_left + tuple(self.get_inverse_relation(r) for r in p_right)[::-1])
                        # right_to_left_paths.add(p_right + tuple(self.get_inverse_relation(r) for r in p_left)[::-1])

        return left_to_right_paths

    # potentially incorrect context paths for which the FINAL entities are not empty
    # i.e. the set of entities that can be reached when traversing r1,...rk starting from e1
    def merge_false_paths_l(self, left_side, right_side, l, potential_false_paths):
        left_l = (l + 1) // 2
        right_l = l // 2
        left_to_right_paths = set()
        left_paths = set(left_side[left_l].keys())
        right_paths = {self.get_inverse_relation(r[0]) for r in right_side[right_l].keys()}
        for path_relations in potential_false_paths:
            p_left = path_relations[0:left_l]
            p_right = path_relations[-1]
            # TODO: maybe we should drop 'p_right in right_paths'?
            if p_left in left_paths and p_right in right_paths:
                last_relation, direction = self.get_composite_relation_index(path_relations[-1], ['in', 'out'])
                valid_path = False
                for entity in left_side[left_l][p_left]:
                    if last_relation in self.knowledge_graph_for_context_paths[entity][direction]:
                        valid_path = True
                        break
                if valid_path:
                    left_to_right_paths.add(path_relations)

        # print('l: potential: {}, actual: {}'.format(len(potential_false_paths), len(left_to_right_paths)))
        return left_to_right_paths

    # potentially incorrect context paths for which the INITIAL entities are not empty
    # i.e. the set of entities that can be reached when traversing rk,...r1 starting from e2
    def merge_false_paths_r(self, left_side, right_side, l, potential_false_paths):
        left_l = l // 2
        right_l = (l+1) // 2
        left_to_right_paths = set()
        left_paths = set(left_side[left_l].keys())
        right_paths = {tuple(self.get_inverse_relation(r_i) for r_i in r)[::-1] for r in right_side[right_l].keys()}
        for path_relations in potential_false_paths:
            p_left = path_relations[0:1]
            p_right = path_relations[1:]
            # TODO: maybe we should drop 'p_left in left_paths'?
            if p_left in left_paths and p_right in right_paths:
                last_relation, direction = self.get_composite_relation_index(path_relations[0], ['out', 'in'])
                valid_path = False
                for entity in right_side[right_l][tuple(self.get_inverse_relation(r_i) for r_i in p_right)[::-1]]:
                    if last_relation in self.knowledge_graph_for_context_paths[entity][direction]:
                        valid_path = True
                        break
                if valid_path:
                    left_to_right_paths.add(path_relations)

        # print('r: potential: {}, actual: {}'.format(len(potential_false_paths), len(left_to_right_paths)))
        return left_to_right_paths

    # yields relation sequences and entities that can be reached over the relation sequences
    def path_traversal(self, path, e_start, e1, e2, max_len, relation, sample, strict_filter=True, single_edge=False):
        for direction in ['in', 'out']:
            relations = list(self.knowledge_graph_for_context_paths[e_start][direction].keys())
            if relation in relations:
                relations.remove(relation)
            previous_relation = path[-1]
            previous_relation_inverse = self.get_inverse_relation(previous_relation)

            for r in relations:
                r_expanded = r
                if direction == 'in':
                    r_expanded = self.get_inverse_relation(r)
                if r_expanded == previous_relation_inverse:
                    continue
                path.append(r_expanded)

                side = 'left' if direction == 'in' else 'right'
                relation_domain = self.relation_domains[r][side]

                if single_edge or not strict_filter or (e1 not in relation_domain and e2 not in relation_domain):
                    possible_entities = self.knowledge_graph_for_context_paths[e_start][direction][r]

                    # if single_edge or strict_filter or (e1 not in possible_entities and e2 not in possible_entities):

                    yield (tuple(path)[1:], possible_entities)

                    if len(path)-1 < max_len:
                        for e_next in possible_entities:
                            yield from self.path_traversal(path, e_next, e1, e2, max_len, relation,
                                                                     sample, strict_filter=strict_filter)

                path.pop()

    '''
    EXPLANATION DATA GENERATION
    '''

    def get_explanation_paths_to_triple(self, triple, sample=True):

        paths = self.get_paths_to_triple(triple, sample)

        if paths is None:
            return None

        known_explanations = self.explanation_correct_paths(triple, paths['paths']['known'])
        unknown_explanations = self.explanation_false_paths(triple, paths['paths']['unknown'])

        explanation_paths = {l: {**known_explanations[l], **unknown_explanations[l]}
                             for l in range(1, self.config['max_context_path_length'] + 1)}
        explanation_data = {'model_input': paths, 'explanation_paths': explanation_paths}

        return explanation_data

    def explanation_false_paths(self, triple, paths):
        e1, _, e2 = triple
        explanation_paths = {l: {} for l in range(1, self.config['max_context_path_length'] + 1)}
        for l in range(1, self.config['max_context_path_length'] + 1):
            for path_relations in paths[l-1]:
                first_relation_norm, direction = self.get_composite_relation_index(path_relations[0], ['in', 'out'])
                if first_relation_norm in self.train_knowledge_graph[e1][direction]:
                    reached_entities_from_left = self.train_knowledge_graph[e1][direction][first_relation_norm]
                else:
                    reached_entities_from_left = {}
                if len(reached_entities_from_left) > 10:
                    reached_entities_from_left = random.sample(reached_entities_from_left, 10)

                last_relation_norm, direction = self.get_composite_relation_index(self.get_inverse_relation(path_relations[-1]),
                                                                                  ['in', 'out'])
                if last_relation_norm in self.train_knowledge_graph[e2][direction]:
                    reached_entities_from_right = self.train_knowledge_graph[e2][direction][last_relation_norm]
                else:
                    reached_entities_from_right = {}
                if len(reached_entities_from_right) > 10:
                    reached_entities_from_right = random.sample(reached_entities_from_right, 10)
                explanation = {'left': reached_entities_from_left, 'right': reached_entities_from_right}
                explanation_paths[l][tuple(path_relations)] = ('false_path', explanation)

        return explanation_paths

    def explanation_correct_paths(self, triple, paths):
        e1, r, e2 = triple
        # skip path length 1 since the entities are just e1 and e2
        explanation_paths = {l: {} for l in range(1, self.config['max_context_path_length'] + 1)}
        for l in range(2, self.config['max_context_path_length']+1):
            for path_relations in paths[l-1]:
                partial_paths = {i: [] for i in range(l+1)}
                partial_paths[0].append([e1])
                for i in range(l):
                    partial_path_set = partial_paths[i]
                    for partial_path in partial_path_set:
                        end_entity = partial_path[-1]
                        next_relation = path_relations[i]
                        next_relation_norm, direction = self.get_composite_relation_index(next_relation, ['in', 'out'])
                        next_entities = self.train_knowledge_graph[end_entity][direction][next_relation_norm] \
                            if next_relation_norm in self.train_knowledge_graph[end_entity][direction] else {}
                        for next_entity in next_entities:
                            extended_path = partial_path + [next_relation, next_entity]
                            partial_paths[i+1].append(extended_path)
                full_paths = partial_paths[len(partial_paths)-1]
                full_paths_filtered = [path for path in full_paths if path[-1] == e2]
                if len(full_paths_filtered) > 10:
                    full_paths_filtered = random.sample(full_paths_filtered, 10)
                explanation_paths[l][tuple(path_relations)] = ('correct_path', full_paths_filtered)

        # no entity sets need to be found for edges
        for path_relations in paths[0]:
            explanation_paths[1][tuple(path_relations)] = ('correct_path', [])

        return explanation_paths

    def explanation_data_writer(self, source_file, output_file, relative_sample_size=None):
        i = 0
        relation_count = {}
        relation_stats = {}
        examples = {}
        skipped = set()

        # collect some stats on relation frequencies
        for _, r, _ in self.data_reader(source_file):
            relation_stats.setdefault(r, 0)
            relation_stats[r] += 1

        with gzip.open(output_file, 'ab') as file:
            for e1, e2, e1_c_domain, e2_c_domain, path in self.single_edge_generator(
                    source_file, parallelize=False, relative_sample_size=relative_sample_size):
                r = path[0]
                triple = e1, r, e2

                relation_count.setdefault(r, 0)

                if relation_count[r] <= 1 and (len(e1_c_domain) > 0 or len(e2_c_domain) > 0):
                    if relation_stats[r] < 2:  # skip rare relations
                        skipped.add(r)
                        continue

                    # corrupt e1 or e2 depending on what is possible, domains may become empty due to type-matching
                    if len(e1_c_domain) > 0 and len(e2_c_domain) > 0:
                        if random.random() > 0.5:
                            corrupted_triple = (random.choice(list(e1_c_domain)), r, e2)
                            description = 'e1_c'
                        else:
                            corrupted_triple = (e1, r, random.choice(list(e2_c_domain)))
                            description = 'e2_c'
                    elif len(e1_c_domain) > 0:
                        corrupted_triple = random.choice(list(e1_c_domain)), r, e2
                        description = 'e1_c'
                    elif len(e2_c_domain) > 0:
                        corrupted_triple = e1, r, random.choice(list(e2_c_domain))
                        description = 'e2_c'
                    else:
                        continue

                    explanation = self.get_explanation_paths_to_triple(triple, sample=False)
                    if explanation is not None:
                        explanation['relation'] = r
                        explanation['description'] = 'fact'
                        explanation['index'] = relation_count[r]
                    else:
                        continue

                    explanation_c = self.get_explanation_paths_to_triple(corrupted_triple, sample=False)
                    if explanation_c is not None:
                        explanation_c['relation'] = r
                        explanation_c['description'] = description
                        explanation_c['index'] = relation_count[r]
                    else:
                        continue

                    examples[r] = (e1, r, e2)
                    pickle.dump(explanation, file)
                    pickle.dump(explanation_c, file)

                    relation_count[r] += 1

                    i += 1
                    if i % 10 == 0:
                        print(i)

            # print('All relations')
            # for relation in relation_stats:
            #     print(relation, relation_stats[relation], self.index.get_relation_name(relation))

            print('\n\nSelected relations')
            for relation in examples:
                print(relation, relation_count[relation], self.index.get_relation_name(relation))

            if len(skipped) > 0:
                print('\n\nSkipped relations')
                for relation in skipped:
                    print(relation, relation_stats[relation], self.index.get_relation_name(relation))

            print('Number of relations in test set: {}'.format(len(relation_stats)))
            print('Number of relations in explanation set: {}'.format(len(examples)))
            print('Total number of facts (correct and corrupted): {}'.format(i))
            print('Number of skipped relations: {}'.format(len(skipped)))

    # generate text for displaying explanations
    @staticmethod
    def get_explanation_output(path_relations, explanations, e1, e2, r, index, dual=True):

        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            ITALIC = '\x1B[3m'
            RED = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        def format_entity(e):
            return bcolors.BOLD + e + bcolors.ENDC

        def format_desc(d):
            return bcolors.ITALIC + d + bcolors.ENDC

        def output_triple(triple):
            e1, r, e2 = index.resolve_names(triple)
            output = dual_output + format_entity(e1) + '  ' + r + '  ' + format_entity(e2) + '\n'
            return output

        def output_entity_set(entities):
            output = ''
            first = True
            for entity in list(entities)[:3]:
                if not first:
                    output += ', '
                output += format_entity(index.get_real_entity_name(entity))
                first = False
            if len(entities) > 3:
                output += ',...'
            if len(entities) > 1:
                return '{' + output + '}'
            else:
                return output

        def output_explanation_false_edges(explanation_path, reached_entities, e1, e2):
            output = ''
            for side in ['left', 'right']:
                if len(reached_entities[side]) > 0:
                    entity_output = output_entity_set(reached_entities[side])

                    output += dual_output
                    if side == 'left':
                        description = format_desc('don\'t include') if len(reached_entities[side]) > 1 \
                            else format_desc('is not the same as')
                        output += format_entity(index.get_real_entity_name(e1)) + '  '\
                                  + index.get_composite_relation_name(explanation_path[0]) + '  '\
                                  + entity_output + '  '\
                                  + description + '  ' \
                                  + format_entity(index.get_real_entity_name(e2)) + '\n'
                    if side == 'right':
                        output += format_entity(index.get_real_entity_name(e1)) + '  ' \
                                  + format_desc('is not') + '  ' \
                                  + entity_output + '  ' \
                                  + index.get_composite_relation_name(explanation_path[0]) + '  ' \
                                  + format_entity(index.get_real_entity_name(e2)) + '\n'
            return output

        def output_explanation_false_paths(explanation_path, reached_entities, e1, e2):
            left_entity_output = output_entity_set(reached_entities['left'])
            right_entity_output = output_entity_set(reached_entities['right'])
            output = dual_output + format_entity(index.get_real_entity_name(e1)) + '  '
            if len(explanation_path) == 2:
                output += index.get_composite_relation_name(explanation_path[0]) + '  ' + left_entity_output + '  '
                if len(reached_entities['left']) == 1 and len(reached_entities['right']) == 1:
                    output += format_desc('is not the same as')
                elif len(reached_entities['left']) > 1 and len(reached_entities['right']) == 1:
                    output += format_desc('don\'t include')
                elif len(reached_entities['left']) == 1 and len(reached_entities['right']) > 1:
                    output += format_desc('does not occur in')
                else:
                    output += format_desc('are different from')
                output += '  ' + right_entity_output + '  ' + index.get_composite_relation_name(explanation_path[1])
            elif len(explanation_path) == 3:
                output += index.get_composite_relation_name(explanation_path[0]) + '  ' + left_entity_output + '  '
                if len(reached_entities['left']) == 1:
                    output += format_desc('has no connection over  ') \
                              + index.get_composite_relation_name(explanation_path[1]) + format_desc('  to  ')
                else:
                    output += format_desc('have no connection over  ') \
                              + index.get_composite_relation_name(explanation_path[1]) + format_desc('  to  ')
                output += right_entity_output + '  ' + index.get_composite_relation_name(explanation_path[2])
            output += '  ' + format_entity(index.get_real_entity_name(e2)) + '\n'
            return output

        def output_explanation_true_paths(explanation_paths):
            output = ''
            sample = False
            if len(explanation_paths) > 3:
                explanation_paths = random.sample(explanation_paths, 3)
                sample = True
            for path in explanation_paths:
                output += dual_output
                for i in range(len(path)):
                    if i % 2 == 0:
                        output += format_entity(index.get_real_entity_name(path[i]))
                    else:
                        output += '  ' + index.get_composite_relation_name(path[i]) + '  '
                output += "\n"
                if path != explanation_paths[-1]:
                    output += "\n"
            if sample:
                output += "...\n"
            return output

        # this is currently not used, as the current CPM does not capture negative correlation,
        # e.g., using incorrect paths that support the correctness of triples
        dual_output = bcolors.RED + 'DUALFACT: ' + bcolors.ENDC if not dual else ''

        if len(path_relations) == 1:
            if path_relations[0] == r:
                return '(the triple itself)\n' + output_triple((e1,) + path_relations + (e2,))
            path_kind, explanation = explanations[len(path_relations)][tuple(path_relations)]
            if path_kind == 'correct_path':
                return output_triple((e1,) + path_relations + (e2,))
            else:
                return output_explanation_false_edges(path_relations, explanation, e1, e2)
            # return output_triple((e1,) + path_relations + (e2,))

        else:
            path_kind, explanation = explanations[len(path_relations)][tuple(path_relations)]
            if path_kind == 'correct_path':
                return output_explanation_true_paths(explanation)
            else:
                return output_explanation_false_paths(path_relations, explanation, e1, e2)

    '''
    FILE READER AND WRITER
    '''

    # Add indices to file names for parallel execution
    def indexed_file(sel, file_path):
        if '.txt' in file_path:
            return file_path.replace('.txt', '_i{}.txt'.format(args.i)) if args.i is not None else file_path
        else:
            return file_path + '_i{}'.format(args.i) if args.i is not None else file_path

    @staticmethod
    def data_reader(file_path):
        with open(file_path, 'r') as file:
            csv_file = csv.reader(file)
            for line in csv_file:
                yield np.array(line).astype(np.int32).tolist()

    def parallel_data_reader(self, file_path, relative_sample_size=None, parallelize=None):
        if parallelize is None:
            if args.i is not None and args.n is not None:
                parallelize = True
            else:
                parallelize = False
        i = -1
        min_index, max_index = self.get_data_partitions(file_path) if parallelize else (0, 0)
        with open(file_path, 'r') as file:
            csv_file = csv.reader(file)
            for line in csv_file:
                i += 1
                if parallelize:
                    if i < min_index:
                        continue
                    if i >= max_index:
                        break
                if relative_sample_size is not None and relative_sample_size < random.random():
                    continue
                yield np.array(line).astype(np.int32).tolist()

    # split data sets into several parts for parallel code execution
    def get_data_partitions(self, data_file):
        data_size = 0
        for _ in self.data_reader(data_file):
            data_size += 1
        min_index = args.i * data_size // args.n
        max_index = (args.i + 1) * data_size // args.n if args.i < args.n - 1 else data_size
        return min_index, max_index

    def write_data_set(self, data, file):
        print('Writing data set')
        with open(file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for line in data:
                writer.writerow(line)

    @staticmethod
    def evaluation_data_reader(file_path):
        def generator():
            for line in DataProcessing.data_reader(file_path):
                e1, e2 = line[0:2]
                e1_domain_size = line[2]
                e2_domain_size = line[3]
                e1_domain = line[4:4 + e1_domain_size]
                e2_domain = line[4 + e1_domain_size:4 + e1_domain_size + e2_domain_size]
                path = line[4 + e1_domain_size + e2_domain_size:]
                yield e1, e2, e1_domain, e2_domain, path

        return generator

    @staticmethod
    def context_path_reader(file_path):
        def generator():
            skipped = 0
            with gzip.open(file_path, 'rb') as file:
                while True:
                    try:
                        yield pickle.load(file)
                    except EOFError:
                        break
                    except pickle.UnpicklingError:
                        skipped += 1
                        continue

                if skipped > 0:
                    print("Skipped because of UnpicklingError: {}".format(skipped))

        return generator

    @staticmethod
    def context_path_explanation_reader(file_path, selected_relations, index):
        def generator():
            with gzip.open(file_path, 'rb') as file:
                while True:
                    try:
                        explanation_data = pickle.load(file)
                        if len(selected_relations) > 0 and explanation_data['relation'] not in selected_relations \
                                or explanation_data['index'] not in index:
                            continue
                        yield explanation_data['model_input']
                    except EOFError:
                        break

        return generator

    '''
    Extract human readable names for mids from the full Freebase data set
    '''

    def generate_real_name_index(self):
        entities = self.index.entity_index['name_to_id'].keys()
        real_name_index = {}
        entities_rdf = {entity[1:].replace('/', '.'): entity for entity in entities}
        # possible relations providing human readable names of mids
        # '<http://rdf.freebase.com/ns/type.object.name>',
        # '<http://rdf.freebase.com/ns/common.topic.alias>',
        # '<http://rdf.freebase.com/key/en>',
        # '<http://rdf.freebase.com/key/wikipedia.en>',
        # '<http://rdf.freebase.com/key/wikipedia.en_title>'
        with gzip.open(self.config['original_database_file'], 'rt') as f:
            found = 0
            while True:
                try:
                    triple = next(f)
                    e1, r, e2, _ = triple.split('\t')
                    if r == '<http://rdf.freebase.com/ns/type.object.name>':
                        e1 = e1[28:-1]
                        if e1 in entities_rdf:
                            if '@en' in e2:
                                e2 = e2[1:-4]
                                # r = r[25:-1]
                                # print(e1, r, e2)
                                real_name_index[entities_rdf[e1]] = e2
                                found += 1
                                if found % 100 == 0:
                                    print(found)
                            elif '@' not in e2:
                                e2 = e2[1:-1]
                                # r = r[25:-1]
                                # print(e1, r, e2)
                                real_name_index[entities_rdf[e1]] = e2
                                found += 1
                                if found % 100 == 0:
                                    print(found)
                except StopIteration:
                    break

        with open(self.config['real_name_index_file'], 'wb') as index_file:
            pickle.dump(real_name_index, index_file)

    '''
    DEBUGGING
    '''

    def print_real_names(self):
        i = 0
        for e1, r, e2 in self.data_reader(self.config['training_file']):
            print(self.index.resolve_names((e1, r, e2)))
            i += 1
        print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', help='Index train, valid and test data', action='store_true')
    parser.add_argument('--plain', help='Single edge data set generation for plain KBC model', action='store_true')
    parser.add_argument('--paths', help='Path data set generation for plain KBC model', action='store_true')
    parser.add_argument('--cpm', help='Data set generation for the CPM', action='store_true')
    parser.add_argument('--explanations', help='Explanation data set generation for the CPM', action='store_true')
    parser.add_argument('--cpm_index', help='Index (and filter) context path-relation combinations', action='store_true')
    parser.add_argument('--update_cpm_index', help='Update cpm index after changing context_path_min_frequency', action='store_true')
    parser.add_argument('--path_relation_stats', help='Show stats on context path-relation combinations', action='store_true')
    parser.add_argument('--real_name_index', help='Extract real names for FreeBase MIDs', action='store_true')
    parser.add_argument('-n', help='Total number of processes for parallel execution', type=int, default=None)
    parser.add_argument('-i', help='Index of process for parallel execution', type=int, default=None)
    parser.add_argument("--config", help="Name of the configuration", default="Params_for_BlackboxNLP")
    parser.add_argument("--verbose", help="More detailed output", action='store_true')
    args = parser.parse_args()
    config = next(config_generator(args.config))
    data = DataProcessing(config)
