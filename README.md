# Context Path Model (CPM)

This repository contains the code of the Context Path Model
and its annotated predictions on FB15K.
Details on this model and experiments conducted with it
can be found in the following paper:

Josua Stadelmaier and Sebastian Padó. 2019. [Modeling Paths for Explainable Knowledge Base Completion](#) (link will follow). In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP.

## Introduction
The CPM generates explanations for new facts in knowledge base completion
by providing sets of context paths as supporting evidence for these triples.
For example, a new triple (Theresa May, nationality, Britain) may be explained
by the path (Theresa May, born in, Eastbourne, contained in, Britain).
The CPM is formulated as a wrapper that can be applied on top of various
existing knowledge base completion models.

In our experiments, we instantiate the CPM with
[TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)
(Bordes et al, 2013) and use the data set [FB15K](https://everest.hds.utc.fr/doku.php?id=en:transe).


## Annotated predictions
We manually evaluated the CPM on identifying paths that provide
the most convincing evidence for or against the correctness of a new triple.
The annotation scheme and the experimental setup is described in the linked paper.

[Annotated predictions with explanations for test triples](explanations/annotated_test_explanations.html)


## Usage
We use Tensorflow 1.12.0 and Python 3.6.5 in our implementation.

### Data processing
1. Index original data set for efficiency (can be skipped for FB15K, index is already provided)
    ```
    $ python data_processing.py --index
    ```
2. Create index of combinations of relations and context paths (already provided for FB15K):
    ```
    $ python data_processing.py --cpm_index
    ```
3. Generate single-edge training and evaluation data for 'plain' KBC models:
    ```
    $ python data_processing.py --plain
    ```
4. Generate context path training and evaluation data for 'plain' KBC models:
    ```
    $ python data_processing.py --paths
    ```
    Path processing can be done on `N` cores in parallel:
    ```
    $ for i in $(seq 0 N-1); do eval "python -u data_processing.py --paths -n N -i $i > logs_$i &"; done
    ```
    After each core has finished, the results can be merged:
    ```
    $ cat data_set_name_i* > data_set_name.txt && rm data_set_name_i*
    ```
5. Generate training and evaluation data for the CPM (can also be parallelized):
    ```
    $ python data_processing.py --cpm
    ```
6. Generate a data set for displaying predictions with explanations:
    ```
    $ python data_processing.py --explanations
    ```

### Training
1. Training plain KBC models on single edges:
    ```
    $ python main.py -d model_description --plain
    ```
2. Training plain KBC models on paths:
    ```
    $ python main.py -d model_description --plain --paths
    ```
3. Training the CPM instantiated with a path-trained KBC model:
    ```
    $ python main.py -d model_description --cpm --paths
    ```

### Testing
- Testing edge-trained KBC models on predicting edges or paths of length `L`:
    ```
    $ python main.py -d model_description --plain --evaluate --path_length L
    ```
- Testing path-trained KBC models on predicting edges or paths of length `L`:
    ```
    $ python main.py -d model_description --plain --paths --evaluate --path_length L
    ```
- Testing the CPM instantiated with a path-trained KBC model on edges:
    ```
    $ python main.py -d model_description --cpm --paths --evaluate
    ```

### Fact prediction and explanation
- Performing fact prediction with explanations using the CPM:
    ```
    $ python main.py -d model_description --cpm --paths --explain --verbose
    ```

## Implementation

### Code:
- `main.py`: Main interface for training and testing.
- `config.py`: All major hyperparameters for reproducing our results and specification of file paths.
- `plain_kbc.py`: Training and evaluation of 'plain' KBC models, e.g. TransE, on edges and paths.
- `cpm.py`: Training and evaluation of the Context Path Model.
- `kbc_model.py`: The parent KBC class for instantiating the CPM with various KBC models.
- `TransE.py`: Child class of kbc_model that implements TransE.
- `data_processing.py`: Data processing for 'plain' KBC models and the CPM.
- `index.py`: Indexing of knowledge bases for efficient processing.


### Directories:
- `data/FB15K/original/`: The original FB15K data set from Bordes et al.
- `data/FB15K/plain_kbc/`: Training and evaluation files for 'plain' KBC models.
- `data/FB15K/cpm/`: Training and evaluation files for the CPM.
- `data/FB15K/index/`: Indices of entity MIDs, entity names, relations and context paths.
- `training_summaries/`: Training summaries for visualization in TensorBoard.
- `checkpoints/`: Saved models.
