# GNN Bill Position Pipeline

This codebase is designed to predict bill positions (edge types) using a Graph Neural Network (GNN).


## Download Input & Output Data
Please download `gnn_input` and `gnn_output` from "Replication Data for: lobbying-position" (https://doi.org/10.7910/DVN/D0QWM2).


## Install

```bash
# Install this as needed depending on your environment
# Since the GNN model can be large, it is recommended to use a `GPU` instead of a `CPU`.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 

pip install termcolor
pip install torch_geometric
pip install lightning
pip install torch_scatter
pip install torch_sparse
```

## Script Overview

#### Graph Configuration
The graph includes multiple types of entities, and different graph configurations are derived from `${gnn_input}`. You can specify the configuration using `${graph_config}`, which corresponds to one of the folder names in the `gnn_input` directory.


## Run GNN
### 1. Run train_gnn.py across hparams

The code below performs a grid search over hyperparameters (e.g., `_num_layers`, `_num_bases`, `_lr`), using the specified GPU `${gpu_num}` to run the GNN.

For a detailed list of hyperparameters, refer to the contents of `train_gnn_label_mapping.sh`.

We set `emb_size` to 90, but if the model doesn't fit on the GPU, you can reduce `emb_size` in `train_gnn_label_mapping.sh` according to your environment.

```bash
bash train_gnn_label_mapping.sh ${gpu_num} \
                                ${gnn_input} \
                                ${graph_config}


e.g.,
# use GPU number 0 and run gnn_input: base 
bash train_gnn_label_mapping.sh 0 \
                                gnn_input 
                                ''

# use GPU number 0 and run gnn_input: base-_legislator 
bash train_gnn_label_mapping.sh 0 \
                                gnn_input \
                                _legislator

# use GPU number 0 and run gnn_input: base-_legislator_lobbyist
bash train_gnn_label_mapping.sh 0 \
                                gnn_input \
                                _legislator_lobbyist
```


### 2. Run train_gnn.py with specific hparams / {N+1} runs

Based on the results of the parameter search above, we run the best setting for each `${graph_config}` N times. The `${nth}` value is passed as a parameter to indicate the current run.

Below is an example of the best setting used in our experiments.
For the full list of hyperparameters, refer to the `train_gnn_label_mapping.sh` script.


```bash
bash train_gnn_label_mapping.sh 0 \
                                gnn_input \
                                _legislator_lobbyist \
                                3 \
                                3 \
                                0.005 \
                                FALSE \
                                FALSE \
                                TRUE \
                                ${nth}
                                
```


### 3. Load best model and predict edge type (bill position)
To obtain the predicted edge types (bill positions) from the **best GNN model**, add the --MODE `load_best_and_predict` flag when running `train_gnn.py`.
With this setting, the script automatically selects the best-performing model from previous runs (based on the given conditions) and uses it to predict the GNN edge types (bill positions).

```bash
# load_best_and_predict
python -W ignore train_gnn.py \
--MODE load_best_and_predict \
--gpu_num ${gpu_num} \
--gnn_input ${gnn_input} \
--dataset base-${graph_config}


e.g.,
python -W ignore train_gnn.py \
--MODE load_best_and_predict \
--gpu_num 0 \
--gnn_input gnn_input \
--dataset base-_legislator_lobbyist
```

If you want to find the best model and its predictions within a more specific environment, you can add more detailed parameters. (The code below is for finding the best model among the settings where num_layers is 3.)

```bash
e.g.,
python -W ignore train_gnn.py \
--MODE load_best_and_predict \
--gpu_num 0 \
--gnn_input gnn_input \
--dataset base-_legislator_lobbyist \
--num_layers 3
```


### 4. Get predicted bill position (edge type)

The predicted model and outputs are saved under the path `gnn_output/_checkpoints/gnn_input+base-[...]/model.pth`. In `gnn_output/load_best_and_predict`, a total of six files are generated, containing `edge_index` and `edge_label_index` for the entire dataset, including the train, validation, and test sets, along with **all edges** in the GNN graph.

