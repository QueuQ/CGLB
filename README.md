![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/logo2.png)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

 <tr><td colspan="4"> <a href="#Get-Started">Get Started</a></td></tr> | <tr><td colspan="4"> <a href="#Dataset-Usages">Dataset Usages</a></td></tr> | <tr><td colspan="4"> <a href="#Pipeline-Usages">Pipeline Usages</a></td></tr> | <tr><td colspan="4"> <a href="#Evaluation-and-Visualization-Toolkit">Evaluation & Visualization Toolkit</a></td></tr> | <tr><td colspan="4"> <a href="#Benchmarks"> Benchmarks </a></td></tr> | <tr><td colspan="4"> <a href="#Acknowledgement"> Acknowledgement </a></td></tr>

 This is the official repository of Continual Graph Learning Benchmark (CGLB), which was published in the datasets and benchmarks track of NeurIPS 2022 [(paper link)](https://openreview.net/forum?id=5wNiiIDynDF). We will keep maintaining this repository to facilitate the development of continual graph learning, and we appreciate any comment on improving our CGLB!

 ```
 @inproceedings{zhang2022cglb,
  title={CGLB: Benchmark Tasks for Continual Graph Learning},
  author={Zhang, Xikun and Song, Dongjin and Tao, Dacheng},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
 ```


 ## Get Started
 
 This repository contains our CGLB implemented for running on GPU devices. To run in the Windows system, please specify the argument ```--replace_illegal_char``` as ```True``` to avoid illegal filename characters (details are in <a href="#Pipeline-Usages">Pipeline Usages</a>). To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6

For GCGL tasks, the following packages are also required:

* rdkit==2020.09.1.0
 
 ## Dataset Usages
 
 ### Importing the Datasets
 For importing the N-CGL datasets, use the following command in python:
 
 ```
 from CGLB.NCGL.utils import NodeLevelDataset
 dataset = NodeLevelDataset('data_name')
```
 
 where the 'data_name' should be replaced by the name of the selected dataset, which are:
 
 ```
 'CoraFull-CL'
 'Arxiv-CL'
 'Reddit-CL'
 'Products-CL'
 ```
 
 For importing the G-CGL datasets, use the following command in python:
 ```
 from CGLB.GCGL.utils import GraphLevelDataset
 dataset = GraphLevelDataset('data_name')
 ```
 
 where the 'data_name' selections are:
 ```
 'SIDER-tIL'
 'Tox21-tIL'
 'Aromaticity-CL'
 ```
 
 
 ### Data Structures and Formats
 
 An instance of the ```NodeLevelDataset``` has multiple attributes and methods. ```dataset.d_data```, ```dataset.n_cls```, and ```dataset.n_nodes``` denotes the number of dimensions of the node features, the number of classes of the corresponding dataset, and the number of nodes in the corresponding dataset, respectively. ```dataset.graph``` is the entire graph of the corresponding dataset without being divided. The data type of ```dataset.graph``` is the [DGL graph](https://docs.dgl.ai/tutorials/blitz/index.html). ```dataset.labels``` contains the label of every node of ```dataset.graph```. ```dataset.tr_va_te_split``` is a dictionary containing the splitting of each class for training, validation, and testing.
 
 ```dataset.get_graph()``` is a most useful method that returns a subgraph of ```dataset.graph``` (for constructing the continual learning task sequence) with the given classes or node indices to retain. There are three keyword arguments for ```dataset.get_graph()``` including ```classes_to_retain```, ```node_ids```, and ```remove_edges```. ```classes_to_retain``` is used to specify which classes of the ```dataset.graph``` to retain in the returned subgraph. ```node_ids``` is used to specify specific nodes to retain. If both ```classes_to_retain``` and ```node_ids``` are specified, their corresponding subgraphs will be separately generated and then combined into one graph. ```remove_edges``` is specifically for removing the edges of the subgraph generated from the specified ```node_ids```. One possible usage of this argument is to select and output the nodes to store for the memory-based baseline ER-GNN.
 
 The usage of the G-CGL dataset resembles classic continual learning. An instance of ```GraphLevelDataset``` consists of four components. ```dataset.dataset``` is the original dataset before being divided. ```train_set, val_set, test_set``` are the splittings for training, validation, and testing. For the task-IL datasets ```SIDER-tIL``` and ```Tox21-tIL```, the original datasets are multi-label ones, and the splitting is simply splitting the dataset into three parts according to the given ratios. While for the ```Aromaticity-CL``` whose original dataset is a multi-class one, the splitting is also done for each class like the N-CGL datasets. As a result, for ```Aromaticity-CL```, each of ```train_set```, ```val_set```, and ```test_set``` is a list containing the splitting of each class. 
 
 ## Pipeline Usages
 
 We provide pipelines for training, validating, and evaluating models with both N-CGL and G-CGL tasks under both task-IL and class-IL scenarios. In the following, we provide several examples to demonstrate the usage of the pipelines.
 ### N-CGL
 Below is the example to run the 'Bare model' baseline with GCN backbone on the Arxiv-CL dataset under the task-IL scenario. 
 
 For both N-CGL and G-CGL experiments, the starting point is the ```train.py``` file, and the different configurations (e.g. the baseline, backbone, learning scenario (task-IL or class-IL), whether to include inter-task edges, etc.) are assigned through the keyword arguments of the Argparse module. For example, to run the N-CGL experiments without inter-task edge under the task-IL scenario, the following code is to be used.
 
 ```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch False \
 ```

For the baselines with tunable hyper-parameters, our framework provide a convenient way to do a grid search over candidate combinations of hyper-parameters. For example, the baseline GEM has two tunable hyper-parameters ```memory_strength``` and ```n_memories```. If the candidate ```memory_strength``` and ```n_memories``` are ```[0.05,0.5,5]``` and ```[10,100,1000]```, respectively, then the command to validate every possible hyper-parameter combination over the validation and test the chosen optimal model on the testing set is as follows:
```
python train.py --dataset Arxiv-CL \
       --gem_args " 'memory_strength': [0.05,0.5,5]; 'n_memories': [10,100,1000] " \
       --method gem \
       --backbone GCN \
       --gpu 0 \
       --ILmode $IL \
       --inter-task-edges $inter \
       --epochs $n_epochs \
```
Some rules to note are: 1. The entire argument ```" 'memory_strength': [0.05,0.5,5]; 'n_memories': [10,100,1000] "``` requires double quotes ``` '' ``` around it. 2. Semicolon ```;``` is required to separate different parameters. 3. Brackets are used to wrap the hyper-parameter candidates. 4. A comma is used to separate the hyper-parameter candidates.
In the example above, all possible nine (three choices for ```memory_strength``` and three choices for ```n_memories```) combinations will be used to set the hyper-parameters.

Since the graphs in N-CGL can be too large to be processed in one batch on most devices, the ```--minibatch``` argument could be specified to be ```True``` for training with the large graphs in mini-batches.
```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch True \
        --batch_size 2000 \
        --sample_nbs True \
        --n_nbs_sample 10,25
 ```
In the above example, besides specifying the ```--minibatch```, the size of each mini-batch is also specified through ```--batch_size```. Moreover, some graphs are extremely dense and will run out the memory even with mini-batch training, which could be addressed through the neighborhood sampling specified via ```--sample_nbs```. And the number of neighbors to sample for each hop is specified through ```--n_nbs_sample```.
There are also other customizable arguments, the full list of which can be found in ```train.py```.

When running the code in the Windows system, the following error **OSError: [Errno 22] Invalid argument** may be triggered and could be avoided by specifying the argument ```--replace_illegal_char``` as ```True``` to replace the potential illegal characters with the underscore symbol ```_```. For example,
 ```
 python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch False \
        --replace_illegal_char True 
 ```

### Modifying the train-validation-test Splitting

The splitting can be simply specified via the arguments when running the experiments. In our implemented pipeline, the corresponding arguments are the validation and testing ratios. For example,

```
python train.py --dataset Arxiv-CL \
        --method bare \
        --backbone GCN \
        --gpu 0 \
        --ILmode taskIL \
        --inter-task-edges False \
        --minibatch False \
        --ratio_valid_test 0.4 0.4
```

The example above set the data ratio for validation and testing as 0.4 and 0.4, and the training ratio is automatically calculated as 0.2.

### Implementing New Methods 

New continual graph learning methods can also be easily implemented in our highly modularized pipelines. 
The newly implemented method should be contained in a Python script file under the directory ```CGLB/NCGL/Baselines```. Suppose we are implementing a method named A, then an ```CGLB/NCGL/Baselines/A_model.py``` containing the implementation of the method should be created. The implementation is flexible as long as it satisfies the input format. Specifically, the Python class of the new method should contain an *observe()* function for model training on a single task, whose input includes the task configurations and the data. Details on the input format could be found in any ```xxx_model.py``` file under the directory ```CGLB/NCGL/Baselines```.

### G-CGL
 Below is an example for running the 'Bare model' baseline with GCN backbone on the SIDER-tIL dataset under the task-IL scenario. 
 ```
 python train.py --dataset $SIDER-tIL \
        --method $Bare \
        --basemodel $GCN \
        --gpu 0 \
        --clsIL False
 ```

 
 ## Evaluation and Visualization Toolkit
 We provide three protocols to evaluate the obtained results as follows. With out pipeline, the results are uniformly stored in the form of a performance matrix, which can be directly fed into our evaluation toolkit.
 
 ### 1. Visualization of the Performance Matrix
 
 This is the most thorough evaluation of a continual learning model since it shows the performance change of each task along the learning process on the entire task sequence. Suppose an experiment result is stored via the path ``` "result_path" ```, the generation of the visualization could be obtained by the following code. Note that the path should be quoted in ``` " " ``` instead of ``` ' ' ```, since ``` ' ' ``` may exist in the file name of the experimental result. Some examples are provided in <tr><td colspan="4"> <a href="#N-CGL-under-Class-IL">N-CGL under Class-IL</a></td></tr>.
 ```
 from CGLB.NCGL.visualize import show_performance_matrices
 show_performance_matrices("result_path")
 ```
 
 ### 2. Learning Curve
 
 This shows the curve of the average performance (AP). It contains less information than the performance matrix but can demonstrate the learning dynamics in a more direct and compact way. Suppose an experiment result is stored via the path ```result_path```, the learning curve could be obtained by the following code. Some examples are provided in <tr><td colspan="4"> <a href="#N-CGL-under-Class-IL">N-CGL under Class-IL</a></td></tr>.
 ```
 from CGLB.NCGL.visualize import show_learning_curve
 show_learning_curve("result_path")
 ```
 
 ### 3. Final AP and Final AF
 Final AP and AF refers to the AP and AF after learning the entire task sequence and is the most compact way to show the performance of a model. Suppose an experiment result is stored via the path ```result_path```, the final AP and AF could be obtained by the following code.
 ```
 from CGLB.NCGL.visualize import shown_final_APAF
 shown_final_APAF("result_path")
 ```
 The outputs with standard deviation are in LaTex form making it easy to copy and paste into a LaTex table.
 
 
 ## Benchmarks
 This section shows our currently obtained results from different baselines. This section will keeps being updated to show state-of-the-art results.
### Dataset Statistics
The statistics of the NCGL datasets are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/NCGL_dataset_statistics.png)

The statistics of the GCGL datasets are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/GCGL_dataset_statistics.png)

### N-CGL under Task-IL
 The results on N-CGL under the task-IL setting without inter-task edges are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/ncgl_taskIL_no_crsedge.png)
 The results on N-CGL under the task-IL setting with inter-task edges are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/ncgl_taskIL_crsedge.png) 

### N-CGL under Class-IL
 The results on N-CGL under the class-IL setting with inter-task edges are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/ncgl_classIL_no_crsedge.png) 
The learning dynamics under the class-IL setting is more meaningful in reflecting the forgetting behavior of the models, therefore, we also show the learning curves and the visualization of the performance matrices as a demonstration of our evaluation & visualization toolkit. The learning curve is obtained on all four datasets, and the performance matrices are visualized on the CoraFull-CL datasets with the largest number of tasks.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/ncgl_classIL_no_crsedge_learning_curve.png) 
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/ncgl_classIL_no_crsedge_performance_matrices_corafull.png) 
 
### G-CGL under Task-IL&Class-IL
 The results on G-CGL under both task-IL and class-IL are shown below.
![CGLB](https://github.com/QueuQ/CGLB/blob/master/figures/gcgl.png) 

 
 ## Acknowledgement
 The construction of CGLB also benefits from existing repositories on both continual learning and continual graph learning. Specifically, the construction of the pipeline for training the continual learning models learns from both [GEM](https://github.com/facebookresearch/GradientEpisodicMemory) and [TWP](https://github.com/hhliu79/TWP). The implementations of the implementations of EWC, GEM learn from [GEM](https://github.com/facebookresearch/GradientEpisodicMemory). The implementations of MAS, Lwf, TWP learn from [MAS](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses) and [TWP](https://github.com/hhliu79/TWP). The implementation of TWP is adapted from [TWP](https://github.com/hhliu79/TWP). The construction of the datasets also benefits from several existing databases and libraries. The construction of the N-CGL datasets uses the datasets and tools from OGB and DGL. The construction of the G-CGL datasets uses the datasets and tools from [DGL](https://docs.dgl.ai/) and [DGL-Lifesci](https://lifesci.dgl.ai/api/data.html).
We sincerely thank the authors of these works for sharing their code and helping developing the community.
