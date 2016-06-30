##
## This script modifies the googlenet prototxts and train the network.
## This script is already being moved to a 'pythonic' and more readable version.
##
## The script takes prepares two runs of the network:
##
## - 1st run:
##      . we set the whole network learning rates to 0, so we are sure the original weights will remain as they are. 
##      . We add a new layer on top of the original network with learning rate != 0. We will teach this layer to map the original network's output features to our labels
## - 2nd run:
##      . We initialize the network with the latest weights from the previous run (so the last layer has been trained). TODO: In the future, we should keep the best snapshot!
##      . We set the learning rate of the whole network to a value != 0. Now the whole network will be adjusted.
##



import subprocess
import os
import sys

sys.path = sys.path + ['./python_datalayers']

PROTOTXT_READY='./data/base_network/my_network/ready_files/ready_network.prototxt'
SOLVER_READY='./data/base_network/my_network/ready_files/ready_solver.prototx'

#Paths where we can find the original files
PROTOTXT_BASE='./base_network/my_network/base_files/googlenetbase.prototxt'
SOLVER_BASE='.PATH_HERE/base_network/my_network/base_files/quick_solver_base.prototxt'

#Path where we will save the snapshots which are going to be produced when training the network
SNAPSHOT_PREFIX='./data/snapshots'

#Path to the folder where we can wait the initial weights in case we want to use some
INITIAL_WEIGHTS='./data/base_network/original/bvlc_googlenet.caffemodel'

#Path to the validation and train filenames
VAL_FILENAME="./data/files/filtered_val.txt";
TRAIN_FILENAME="./data/files/filtered_train.txt";


# We check how many classes there are in the output. Please, be sure when you build the dataset that the classes are numbered from 0 to (OUTPUT_CLASSES - 1)
OUTPUT_CLASSES= int(subprocess.check_output("cat {train_filename}  | cut -d ',' -f2 | tr ' ' '\n' | sort | uniq | wc -l", shell = True).format(train_filename = TRAIN_FILENAME))
os.system('mkdir -p ./data/base_network/my_network/ready_files')

######################################################
# 1st run
LR_MULT_LAST_LAYER=1;
DE_MULT_LAST_LAYER=2;
LR_MULT_BASE_NET=0;
DE_MULT_BASE_NET=0;
ITERS=2500;

OUTPUT_AUX=$(cat $PROTOTXT_BASE | sed "s|loss1/classifier|loss1/classifier_new|g"  | sed "s|loss2/classifier|loss2/classifier_new|g"  | 
	sed "s|loss3/classifier|loss3/classifier_new|g" | sed "s|OUTPUT_NEURONS|$OUTPUT_CLASSES|g")
OUTPUT_AUX=$(echo $OUTPUT_AUX | sed "s|LR_MULT_BASE_NET|$LR_MULT_BASE_NET|g" | sed "s|DE_MULT_BASE_NET|$DE_MULT_BASE_NET|g" | 
								sed "s|LR_MULT_LAST_LAYER|$LR_MULT_LAST_LAYER|g" | sed "s|DE_MULT_LAST_LAYER|$DE_MULT_LAST_LAYER|g")
OUTPUT_AUX=$(echo $OUTPUT_AUX | sed "s|VAL_FILENAME|$VAL_FILENAME|g"  | sed "s|TRAIN_FILENAME|$TRAIN_FILENAME|g")


echo $OUTPUT_AUX > $PROTOTXT_READY
cat $SOLVER_BASE | sed "s|ITERS|$ITERS|g" | sed "s|SNAPSHOT_PREFIX|$SNAPSHOT_PREFIX/snapshot_stage_1|g" | sed "s|MODEL_TO_TRAIN|$PROTOTXT_READY|g"> $SOLVER_READY;
#/home/ubuntu/caffenew/build/tools/caffe train -solver $SOLVER_READY -weights $INITIAL_WEIGHTS 2> $PATH_HERE/data/logs/train_stage.error > $PATH_HERE/data/logs/train_stage.log;

echo "Fine tunning"

##Load last snapshot and keep running
last_snapshot=$(cat data/logs/train_stage.error | grep "Snapshotting to binary proto file" | tail -n 1 | cut -d ' ' -f10)
echo $last_snapshot

######################################################
#2nd run
LR_MULT_LAST_LAYER=0.5;
DE_MULT_LAST_LAYER=1;
LR_MULT_BASE_NET=0.5;
DE_MULT_BASE_NET=1;
ITERS=1000000;
OUTPUT_AUX=$(cat $PROTOTXT_BASE | sed "s|loss1/classifier|loss1/classifier_new|g"  | sed "s|loss2/classifier|loss2/classifier_new|g"  | sed "s|loss3/classifier|loss3/classifier_new|g" | sed "s|OUTPUT_NEURONS|$OUTPUT_CLASSES|g")
OUTPUT_AUX=$(echo $OUTPUT_AUX | sed "s|LR_MULT_BASE_NET|$LR_MULT_BASE_NET|g" | sed "s|DE_MULT_BASE_NET|$DE_MULT_BASE_NET|g" | sed "s|LR_MULT_LAST_LAYER|$LR_MULT_LAST_LAYER|g" | sed "s|DE_MULT_LAST_LAYER|$DE_MULT_LAST_LAYER|g")
OUTPUT_AUX=$(echo $OUTPUT_AUX | sed "s|VAL_FILENAME|$VAL_FILENAME|g"  | sed "s|TRAIN_FILENAME|$TRAIN_FILENAME|g")
echo $OUTPUT_AUX > $PROTOTXT_READY
cat $SOLVER_BASE | sed "s|ITERS|$ITERS|g" | sed "s|SNAPSHOT_PREFIX|$SNAPSHOT_PREFIX/snapshot_stage_2|g" | sed "s|MODEL_TO_TRAIN|$PROTOTXT_READY|g"> $SOLVER_READY;
#/home/ubuntu/caffenew/build/tools/caffe train -solver $SOLVER_READY -weights $last_snapshot   2> $PATH_HERE/data/logs/train_stage_fine.error > $PATH_HERE/data/logs/train_stage_fine.log;


######################################################
#Test
last_snapshot=$(cat data/logs/train_stage.error | grep "Snapshotting to binary proto file" | tail -n 1 | cut -d ' ' -f10)
/home/ubuntu/caffenew/build/tools/caffe test -model $PROTOTXT_READY -weights $last_snapshot
