What networks contains this repository:
---------------------------------------

There are two networks implemented here. A monolabel network and a multilabel network

- Multilabel network: in stand by. The last requirements seems to be that the network must be parallelized and this multilabel network is not. Because of this the task force has been focused on the monolabel network.

- Monolabel network: this network is fully functional at the moment. Each frame is analyzed separately and a top classifier predicts the tags for the whole video taking into account the predicted labels for each frame.

Source code structure:
----------------------

- experiments: contains each one of the network. Each folder inside experiments must implement all the required steps to train and test the network. These folders must also include the prototxts templates for the networks they implement.
- python_datalayers: implementation of python layers and some other utilities. The code in this folder must be network-agnostic even when they are implemented thinking in a specific architecture.
- PythonNetwork: contains pythonic code to run Caffe. At the moment it only includes code for the forward pass. This code is network agnostic.
- template_tools: contains code responsible of parsing and filling prototxts templates. Each experiment can use these tools to manage their templates. The idea behind this is to avoid maintaining train and deploy prototxts. Hopefully Caffe will solve this eventually, but atm these templates are quite useful (ugly, though... I know). 

Experiment structure and instructions to run the code:
------------------------------------------------------

First, choose what experiment you are going to run. Let's see the experiment is named as EXPERIMENT_NAME
All the code require to work with this network can be found at: experiments/$EXPERIMENT_NAME

Inside a specific experiment folder you can find the following folders:

- base_network: prototxts used to build the network. Can be both prototxts or templates to be filled later. These files will never be overwritten by the rest of the code
- data: inside here you can find many folders where we save both files required by the network and outputs. It includes snapshots, logs, datasets, prototxts (remember that you might have templates, here we write the final one).
- source: all the scripts you need to execute this network

The steps required are as folows:

0) Write the settings.py file. PLEASE, DO IT! Specifically, try to set a specific name for your experiment. This will allow us to identify which files have been generated during your specific execution.

1) We need to prepare the dataset given the input files. This code will transform the labels, will chose only the classes with more than X occurrences and will prepare the training files. It will also write a file to map back classes to labels. The specific paths can be found in 'settings.py'

  PYTHONPATH=$PYTHONPATH:python_datalayers:.:experiments/$EXPERIMENT_NAME/source python experiments/$EXPERIMENT_NAME/source/_1_a_split_train_test.py 

2) We split the data in train and test. We check that all the frames of a specific video are in the same group (test or train) to avoid any overfitting.

  PYTHONPATH=$PYTHONPATH:python_datalayers:.:experiments/$EXPERIMENT_NAME/source python experiments/$EXPERIMENT_NAME/source/_1_b_split_train_test.py 

3) We train the network.

  PYTHONPATH=$PYTHONPATH:python_datalayers:.:experiments/$EXPERIMENT_NAME/source python experiments/$EXPERIMENT_NAME/source/_2_train_process.py

4) We make a forward pass on the network. We save the outputs to get some metrics

  PYTHONPATH=$PYTHONPATH:python_datalayers:.:experiments/$EXPERIMENT_NAME/source python experiments/$EXPERIMENT_NAME/source/_3_test_and_save_outputs.py 

5) We get our metrics after implementing a classifier on top.

  PYTHONPATH=$PYTHONPATH:python_datalayers:.:experiments/$EXPERIMENT_NAME/source python experiments/$EXPERIMENT_NAME/source/_4_analyze_results.py 


