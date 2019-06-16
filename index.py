import csv
import numpy as np
import pickle

'''
This class handles indexing of knowledge bases for efficient data processing.
It also provides methods for displaying human understandable names
of entities and relations.
'''


class Index(object):
    def __init__(self, config, entity_index, relation_index):
        self.config = config
        self.entity_index = entity_index
        self.relation_index = relation_index
        self.entity_count = len(self.entity_index['id_to_name'])
        self.relation_count = len(self.relation_index['id_to_name'])
        with open(self.config['real_name_index_file'], 'rb') as file:
            self.real_names = pickle.load(file)

    @staticmethod
    def load(config):
        entity_index = Index.load_index_from_file(config['entity_index_file'])
        relation_index = Index.load_index_from_file(config['relation_index_file'])
        return Index(config, entity_index, relation_index)

    @staticmethod
    def create(config):
        entities, relations = Index.get_domains(config)
        entity_index = Index.create_index_from_data(entities, config['entity_index_file'])
        relation_index = Index.create_index_from_data(relations, config['relation_index_file'])
        return Index(config, entity_index, relation_index)

    @staticmethod
    def get_domains(config):
        with open(config['original_training_file'], 'r') as file:
            entities = set()
            relations = set()

            for line in csv.reader(file, delimiter='\t'):
                e1, r, e2 = line[:3]
                relations |= {r}
                entities |= {e1, e2}

        return entities, relations

    @staticmethod
    def create_index_from_data(index_elements, index_file):
        with open(index_file, 'w') as csv_file:
            index = {'id_to_name': {}, 'name_to_id': {}}
            writer = csv.writer(csv_file)
            for id, name in enumerate(index_elements):
                index['id_to_name'][np.uint32(id)] = name
                index['name_to_id'][name] = np.uint32(id)
                writer.writerow([id, name])
        return index

    @staticmethod
    def load_index_from_file(index_file):
        index = {'id_to_name': {}, 'name_to_id': {}}
        with open(index_file, 'r') as file:
            for id, name in csv.reader(file):
                index['id_to_name'][np.uint32(id)] = name
                index['name_to_id'][name] = np.uint32(id)
        return index

    def apply(self, t):
        e1, r, e2 = t
        e1 = self.entity_index['name_to_id'][e1]
        e2 = self.entity_index['name_to_id'][e2]
        r = self.relation_index['name_to_id'][r]
        return e1, r, e2

    def resolve_names(self, t):
        e1, r, e2 = t
        e1 = self.entity_index['id_to_name'][e1]
        if e1 in self.real_names:
            e1 = self.real_names[e1]
        e2 = self.entity_index['id_to_name'][e2]
        if e2 in self.real_names:
            e2 = self.real_names[e2]
        r = self.get_relation_name(r) if r < self.relation_count else '-' + self.get_relation_name(r - self.relation_count)
        return e1, r, e2

    def get_real_entity_name(self, e):
        e = self.entity_index['id_to_name'][e]
        if e in self.real_names:
            e = self.real_names[e]
        return e

    def get_composite_relation_name(self, r):
        return self.get_relation_name(r) if r < self.relation_count else '-' + self.get_relation_name(r - self.relation_count)

    def get_entity_name(self, e):
        e = self.entity_index['id_to_name'][e]
        return e

    def get_relation_name(self, r):
        r = self.relation_index['id_to_name'][r]
        return r

    def resolve_path_names(self, path):
        relation_names = [self.get_relation_name(r) if r < self.relation_count
                          else '-' + self.get_relation_name(r - self.relation_count) for r in path[1:-1]]

        e1_name = self.get_entity_name(path[0])
        if e1_name in self.real_names:
            e1_name = self.real_names[e1_name]

        e2_name = self.get_entity_name(path[-1])
        if e2_name in self.real_names:
            e2_name = self.real_names[e2_name]

        return [e1_name] + relation_names + [e2_name]

    def resolve_relation_path_names(self, path):
        relation_names = [self.get_relation_name(r) if r < self.relation_count
                          else '-' + self.get_relation_name(r - self.relation_count) for r in path]
        return relation_names

