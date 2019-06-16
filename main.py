from config import config_generator
from plain_kbc import KBC
from cpm import CPM
import argparse

'''
This is the main interface for training and evaluating
plain KBC models and the CPM.

When lists of hyperparameters are given in config.py
and not otherwise specified with -b, -e or -i,
all possible hyperparameter combinations will be executed.
'''


# choose action based on parameters
def run(config):
    if args.plain:
        if args.evaluate:
            KBC(config, args).evaluate()
        else:
            KBC(config, args).train()
    elif args.cpm:
        if args.evaluate:
            CPM(config, args).evaluate_online()
        elif args.explain:
            CPM(config, args).predict_and_explain()
        elif args.analyse:
            CPM(config, args).analyse_relevance_scores()
        else:
            CPM(config, args).train()


def print_config(config):
    print('\nConfiguration {}:'.format(config['id']))
    for param, value in config.items():
        if 'file' not in param and 'dir' not in param:
            if 'print' not in config or param in config['print']:
                print('{}: {}'.format(param, value))


parser = argparse.ArgumentParser()
parser.add_argument("--plain", help="Train the plain KBC model.", action="store_true")
parser.add_argument("--paths", help="Path training", action="store_true")
parser.add_argument("--cpm", help="Train the context aware model", action="store_true")
parser.add_argument("--evaluate", help="Evaluate model", action="store_true")
parser.add_argument("--explain", help="Predict and explain using context-aware model", action="store_true")
parser.add_argument("--config", help="Name of the configuration", default="Params_for_BlackboxNLP")
parser.add_argument("-b", help="Start index of configurations", type=int, default=0)
parser.add_argument("-e", help="End index of configurations", type=int, default=None)
parser.add_argument("-i", help="Index of configuration", type=int, default=None)
parser.add_argument("--part", help="Partition of the evaluation files", type=int, default=None)
parser.add_argument("-d", help="Description of configuration", default="")
parser.add_argument("-cd", help="Description of configuration for context-aware model", default="")
parser.add_argument("--joint_training", help="Joint training of plain KBC model and CPM", action="store_true")
parser.add_argument("--restore", help="Restore model from checkpoint, config has to match", action="store_true")
parser.add_argument("--path_length", help="Path length for evaluation", type=int, default=None)
parser.add_argument("--gpu", help="Index of used gpu", default=None)
parser.add_argument("--corrupted_entity", help="Which entity to corrupt during evaluation", default="")
parser.add_argument("--analyse", help="Analyse relevance scores of paths-relation combinations.", action="store_true")
parser.add_argument("--list_configs", help="List all configuration combinations", action="store_true")
parser.add_argument("--verbose", help="Print explanations.", action="store_true")
parser.add_argument("--checkpoint", help="The checkpoint file used for restoring a saved model", default="")
parser.add_argument("--valid", help="Use validation file for CPM evaluation", action="store_true")
parser.add_argument("--annotate", help="Annotation mode", action="store_true")
parser.add_argument("--filter_annotated", help="Hide explanations that are already annotated", action="store_true")
parser.add_argument("--context_only", help="Do not use the triple at all, only the context paths.", action="store_true")
args = parser.parse_args()


# generate all possible configurations of hyperparameters
configurations = list(config_generator(args.config))

# select one configuration
if args.i is not None:
    config = configurations[args.i]
    config['id'] = args.i
    if not args.evaluate:
        print_config(config)
    run(config)

# run several configurations
else:
    args.e = len(configurations) if args.e is None else args.e
    for i in range(args.b, args.e):
        configurations[i]['id'] = i
        print_config(configurations[i])
        if args.list_configs:
            continue
        run(configurations[i])


