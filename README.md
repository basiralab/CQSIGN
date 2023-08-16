# CQSIGN: Topological Contracted Quantized SIGN Framework
This repository is part of "Affordable Graph Neural Network Framework using Topological Graph Contraction" workshop paper published in MICCAI workshop ([MiLLAND 2023](https://miccaimilland.wixsite.com/milland2023)). This repository contains the main source code for our proposed topological graph contraction and its application on various memory-efficient GNN models (e.g., SIGN, QSIGN, GCN, Cluster-GCN, and GNN AutoScale). However, since some of the included model have different dependencies (e.g., different version of NetworkX), this repository is organized as follows:

<img width="600" alt="Screenshot 2023-08-14 at 13 25 44" src="https://github.com/christopheradnel414/CQSIGN/assets/41734037/56969b5b-258e-4150-98fe-3c1984a5982a">

The main directory of CQSIGN consists of our proposed centrality-based topological graph contraction and the implementation of C-QSIGN, C-SIGN, and C-GCN models (including the [ActNN](https://github.com/ucbrise/actnn) quantization package). Additionally, the main directory also has 2 data generator scripts which generates contracted graphs of our datasets (e.g., PPI, Organ-C, and Organ-S) in a format that is readable to [Cluster-GCN](https://github.com/google-research/google-research/tree/master/cluster_gcn) and [GNN AutoScale (GAS)](https://github.com/rusty1s/pyg_autoscale). Hence, once the graph has been contracted, the included Cluster-GCN and GAS models can be executed with their respective original dependency, independent of our graph contraction module.

# Dependencies
Here, we have 3 different sets of package dependencies for the main CQSIGN directory (includes C-QSIGN, C-SIGN, C-GCN), Cluster-GCN directory, and GNN AutoScale directory, respectively. For the CQSIGN directory, we worked the following dependencies:
```
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-geometric==2.3.0
torchvision==0.15.1+cu118
torch==2.0.0+cu118
ninja==1.11.1
numpy==1.24.1
networkx==3.0
scikit-learn==1.2.2
scipy==1.10.1
```
Next, for the included Cluster-GCN package, we worked with the following dependencies:
```
nvidia-tensorflow==1.15.5+nv22.12
metis==0.2a5
numpy==1.23.5
networkx==1.11
scikit-learn==1.2.2
scipy==1.10.1
setuptools
```
Finally, for the included GNN AutoScale package, we worked with the following dependencies:
```
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-geometric==2.3.0
torch==2.0.0+cu118
metis==0.2a5
hydra-core==1.3.2
numpy==1.24.1
scikit-learn==1.2.2
scipy==1.10.1
setuptools
```
Note that we are using CUDA 11.8 toolkit paired with an NVIDIA RTX 3050ti GPU with driver version 525.125.06. User are recommended to use the CUDA toolkit version that corresponds to their NVIDIA GPU driver. Details can be found [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).

# Setup
Note that these instructions are written for Linux (Ubuntu 22.04 LTS) with NVIDIA driver version 525.125.06.
## CQSIGN Directory Setup
1. To setup the main CQSIGN directory, user is recommended to create a new Python 3.9.16 virtual environment using [conda](https://conda.io/projects/conda/en/latest/index.html).
3. Install NVIDIA CUDA 11.8 toolkit from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive). Depending on the user's NVIDIA driver version, different version of CUDA toolkit might be necessary.
4. Add the newly installed CUDA toolkit directory to bashrc by adding these lines to ~/.bashrc file:
```
# CUDA Toolkit 11.8
if [ -z $LD_LIBRARY_PATH ]; then
  LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
else
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
fi
export LD_LIBRARY_PATH
export PATH="/usr/local/cuda-11.8/bin:$PATH"
```
4. Install dependencies using the following pip commands:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+11.8.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+11.8.html
pip install torch-geometric
pip install ninja==1.11.1
pip install numpy==1.24.1
pip install networkx==3.0
pip install scikit-learn==1.2.2
pip install scipy==1.10.1
```
5. Install ActNN quantization package by going to ActNN folder and running the following script in cli:
```
pip install -v -e .
```
## Cluster-GCN Directory Setup
1. User is recommended to create a new Python 3.9.16 virtual environment for the Cluster-GCN dependencies.
2. Go to CQSIGN/OtherBenchmarks/Cluster-GCN/GKlib and install [GKlib](https://github.com/KarypisLab/GKlib) by executing the following script:
```
make
make install
```
3. CQSIGN/OtherBenchmarks/Cluster-GCN/METIS and install [METIS](https://github.com/KarypisLab/METIS) by executing the following script:
```
sudo apt-get install build-essential
sudo apt-get install cmake

make config shared=1 cc=gcc prefix=~/local
make install
```
4. Install python metis wrapper using pip:
```
pip install metis
```
5. Set METIS_DLL environment variable by adding the following script to ~/.bashrc:
```
export METIS_DLL=~/.local/lib/libmetis.so
```
6. Install dependencies using the following pip commands:
```
pip install nvidia-tensorflow
pip install networkx==1.11
pip install numpy==1.23.5
pip install scikit-learn==1.2.2
pip install scipy==1.10.1
pip install setuptools
```
## GNN AutoScale (GAS) Directory Setup
1. Here, user can reuse the same conda environment as the main CQSIGN directory as there are no conflicting dependencies with GNN AutoScale.
2. Install METIS from step 2, 3, 4, and 5 of Cluster-GCN Directory Setup if not done before.
3. Install remaining dependancies using pip:
```
pip install hydra-core==1.3.2
pip install setuptools
```
4. Go to CQSIGN/OtherBenchmarks/GNNAutoScale/pyg_autoscale and compile the C++ files using the following script:
```
python setup.py install
```

# Executing Benchmarks

## Preparing Dataset
In this work, we are using 3 main datasets ([PPI](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.PPI.html), [MedMNIST Organ-C](https://medmnist.com/), and [MedMNIST Organ-S](https://medmnist.com/)). PPI dataset will automatically downloaded from the installed torch-geometric module when we run the benchmarks, while Organ-C & S datasets are included in this github repository. However, since Organ-C & S are originally an image dataset, we included a pre-processing script to convert the image to graph based on their vectorized pixel intensities and between sample cosine similarity. Before initial run of the code, user needs to run the Organ pre-processing script as follows:
```
./generate_organ.sh
```

## Pre-processing for ClusterGCN and GNN AutoScale
As mentioned in the introduction, both included ClusterGCN and GNN AutoScale package takes the pre-processed contracted graph dataset as input. Hence, we need to generate these contracted graphs using the following script:
```
./run_generate_contracted_data_ClusterGCN.sh
./run_generate_contracted_data_GNNAutoScale.sh
```

## Executing C-QSIGN, C-SIGN, and C-GCN
To reproduce the experiments for C-QSIGN, C-SIGN, and C-GCN, the user can run the following script:
```
./run_main_benchmark_GCN_SIGN_QSIGN.sh
```
To manually run the experiment with user defined parameters, the user can use the following command:
```
python main_{dataset name}.py --model_type {model name} --centrality {centrality type} --num_epoch {number of epoch} --node_budget {node budget} --max_hop {SIGN maximum hop} --layers {number of layers} --dropout {dropout} --batch_norm {batch normalization} --lr {learning rate} --num_batch {number of mini-batches}
```
By setting the parameters as follows:
- {dataset name}: a choice between PPI, OrganC, and OrganS
- {model name}: a choice between GCN, SIGN, and QSIGN
- {centrality type}: a choice between NO (no contraction), DC (degree centrality), EC (eigenvector centrality), BC (betweenness centrality), PR (PageRank centrality), and CC (closeness centrality), which denotes the centrality measure being used for the contraction
- {number of epoch}: integer denoting the number of epoch
- {node budget}: integer denoting the number of training node budget post-contraction
- {SIGN maximum hop}: integer denoting the number of SIGN and QSIGN pre-computed hops (corresponds to the maximum adjacency power matrix)
- {number of layers}: integer denoting the number of layers
- {dropout}: float between 0 to 1.0 denoting the dropout
- {batch normalization}: boolean denoting whether or not to use batch normalization (True: use batchnorm, False: do not use batchnorm)
- {learning rate}: float denoting the learning rate
- {number of mini-batches}: integer denoting the number of mini-batches for the training

Example: 
```
python main_PPI.py --model_type QSIGN --centrality EC --num_epoch 1000 --node_budget 15000 --max_hop 3 --layers 3 --dropout 0.2 --batch_norm True --lr 0.0005 --num_batch 10
```
Note that the results of the run will be appended to results.csv file (will automatically be created if it doesn't exists) in a tabular manner.
## Executing C-ClusterGCN
To run the C-ClusterGCN experiment, the user must first go to this directory: CQSIGN/OtherBenchmarks/Cluster-GCN/cluster_gcn
Next, the user can reproduce the benchmarks using the following script:
```
./run_clusterGCN_benchmark.sh
```
Note that the results of the run will be appended to results.csv file (will automatically be created if it doesn't exists) in a tabular manner.
## Executing C-GAS (GNN AutoScale)
To run the C-GAS experiment, the user must first go to this directory: CQSIGN/OtherBenchmarks/GNNAutoScale/pyg_autoscale/large_benchmark
Next, the user can reproduce the benchmarks using the following script:
```
./run_GAS_benchmark.sh
```
Note that the results of the run  be appended to results.csv file ( automatically be created if it doesn't exists) in a tabular manner.

