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

import template_tools
from template_tools.template_manager1 import PrototxtTemplate
import subprocess
import os
import sys
from settings import settings


def trainNetworkFromScratch(CLASSIFIER_NAME, OUTPUT_CLASSES, VAL_FILENAME, TRAIN_FILENAME, map_template2file):

    #Paths where we can find the original files
    PROTOTXT_BASE=settings['PROTOTXT_BASE']
    SOLVER_BASE=settings['SOLVER_BASE']

    #Path to the folder where we can wait the initial weights in case we want to use some
    INITIAL_WEIGHTS=settings['INITIAL_WEIGHTS']

    PROTOTXT_READY=settings['PROTOTXT_READY']
    SOLVER_READY=settings['SOLVER_READY']

    #Path where we will save the snapshots which are going to be produced when training the network
    SNAPSHOT_PREFIX=settings['SNAPSHOT_PREFIX']



    # We check how many classes there are in the output. Please, be sure when you build the dataset that the classes are numbered from 0 to (OUTPUT_CLASSES - 1)

    os.system('mkdir -p ' + settings['PROTOTXT_READY_PATH'])

    #########################################################
    ###------------------ 1st stage ----------------------###
    #########################################################
    last_snapshot = INITIAL_WEIGHTS

    variables_to_replace = {
        'LRMULTBASENET' : '0',
        'DEMULTBASENET' : '0',
        'LRMULTLASTLAYER' : '1',
        'DEMULTLASTLAYER' : '2',
        'OUTPUTNEURONS' : str(OUTPUT_CLASSES),
        'TRAINFILENAME': TRAIN_FILENAME,
        'VALFILENAME': VAL_FILENAME
    }

    new_prototxt = PrototxtTemplate(PROTOTXT_BASE, map_template2file)
    new_prototxt.saveOutputPrototxt(PROTOTXT_READY, variables_to_replace)

    variables_to_replace = {
        'ITERS' : '20', #'2000',
        'SNAPSHOTPREFIX' : SNAPSHOT_PREFIX + '/%s_snapshot_stage_1' % CLASSIFIER_NAME,
        'MODELTOTRAIN': PROTOTXT_READY
    }

    new_solver_prototxt = PrototxtTemplate(SOLVER_BASE, {})
    new_solver_prototxt.saveOutputPrototxt(SOLVER_READY, variables_to_replace)

    os.system('/home/ubuntu/caffenew/build/tools/caffe train -solver {SOLVER_READY} -weights {INITIAL_WEIGHTS} 2> {b}/{a}_train_stage_1.error > {b}/{a}_train_stage_1.log'.format(SOLVER_READY = SOLVER_READY, INITIAL_WEIGHTS = last_snapshot, a = CLASSIFIER_NAME, b = settings['LOGS_PATH']))



    #########################################################
    ###------------------ 2st stage ----------------------###
    #########################################################
    snapshot_prefix_looked_for = '%s_snapshot_stage_1' % CLASSIFIER_NAME
    last_snapshot=settings['SNAPSHOT_PREFIX'] + "/" + sorted([(int(x.split(".")[0]), name) for x,name in [(x.split("_")[-1],x) for x in os.listdir(settings['SNAPSHOT_PREFIX']) if 'caffemodel' in x and snapshot_prefix_looked_for in x]], key = lambda x: x[0], reverse = True)[0][1]

    variables_to_replace = {
        'LRMULTBASENET' : '1',
        'DEMULTBASENET' : '0.5',
        'LRMULTLASTLAYER' : '1',
        'DEMULTLASTLAYER' : '0.5',
        'OUTPUTNEURONS' : str(OUTPUT_CLASSES),
        'TRAINFILENAME': TRAIN_FILENAME,
        'VALFILENAME': VAL_FILENAME
    }


    new_prototxt = PrototxtTemplate(PROTOTXT_BASE, map_template2file)
    new_prototxt.saveOutputPrototxt(PROTOTXT_READY, variables_to_replace)

    variables_to_replace = {
        'ITERS' : '20',#'5000',
        'SNAPSHOTPREFIX' : SNAPSHOT_PREFIX + '/%s_snapshot_stage_2' % CLASSIFIER_NAME,
        'MODELTOTRAIN': PROTOTXT_READY
    }

    new_solver_prototxt = PrototxtTemplate(SOLVER_BASE, {})
    new_solver_prototxt.saveOutputPrototxt(SOLVER_READY, variables_to_replace)

    os.system('/home/ubuntu/caffenew/build/tools/caffe train -solver {SOLVER_READY} -weights {INITIAL_WEIGHTS} 2> {b}/{a}_train_stage_2.error > {b}/{a}_train_stage_2.log'.format(a = CLASSIFIER_NAME, SOLVER_READY = SOLVER_READY, INITIAL_WEIGHTS = last_snapshot, b = settings['LOGS_PATH']))



if __name__ == '__main__':
    CLASSIFIER_NAME = settings['experiment_name']

    #Path to the validation and train filenames
    VAL_FILENAME="./data/files/filtered_val.txt"
    TRAIN_FILENAME="./data/files/filtered_train.txt"

    command = "cat {train_filename}  | cut -d ',' -f2 | tr ' ' '\n' | sort | uniq | wc -l".format(train_filename = TRAIN_FILENAME)
    OUTPUT_CLASSES= int(subprocess.check_output(command, shell = True))

    print("Training network with name " + CLASSIFIER_NAME + " which has " + str(OUTPUT_CLASSES) + ' classes')
    
    trainNetworkFromScratch(CLASSIFIER_NAME, OUTPUT_CLASSES, VAL_FILENAME, TRAIN_FILENAME, settings['map_template2file']['TRAIN'])
