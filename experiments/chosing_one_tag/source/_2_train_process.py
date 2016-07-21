
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
from _stupid_tools_and_helpers_scripting import _aux_getNumberOfCasses, _aux_getSnapshotToBeused, _getPredefinedVariables_
import subprocess
import os
import sys
from settings import settings

def trainStage(initial_weights, output_classes, last_layer_lr_mult, rest_layers_lr_mult, iters, name_for_stage):
    last_snapshot = initial_weights
    ## 0 - We remove the existing snapshots with this name, so we will not take the wrong one
    removeExistingSnapshots(name_for_stage)

    ## 1 - Let's create prototxt network based on our templates
    variables_to_replace = {
        'LRMULTBASENET' : str(1*rest_layers_lr_mult),
        'DEMULTBASENET' : str(0.5*rest_layers_lr_mult),
        'LRMULTLASTLAYER' : str(1*last_layer_lr_mult),
        'DEMULTLASTLAYER' : str(0.5*last_layer_lr_mult),
        'OUTPUTNEURONS' : str(output_classes),
        'TRAINFILENAME': settings['output_file_train'],
        'VALFILENAME': settings['output_file_test']
    }
    new_prototxt = PrototxtTemplate(settings['PROTOTXT_BASE'], settings['map_template2file']['TRAIN'])
    new_prototxt.saveOutputPrototxt(settings['PROTOTXT_READY'], variables_to_replace)

    ## 2 - Now we create the solver
    variables_to_replace = {
        'ITERS' : str(iters), #'2000',
        'SNAPSHOTPREFIX' : settings['SNAPSHOT_PREFIX'] + '/%s_%s' % (settings['experiment_name'], name_for_stage),
        'MODELTOTRAIN': settings['PROTOTXT_READY']
    }
    new_solver_prototxt = PrototxtTemplate(settings['SOLVER_BASE'], {})
    new_solver_prototxt.saveOutputPrototxt(settings['SOLVER_READY'], variables_to_replace)

    # 3 - Now we execute the network
    os.system('/home/ubuntu/caffenew/build/tools/caffe train -solver {SOLVER_READY} -weights {INITIAL_WEIGHTS} 2> {b}/{a}_{c}.error > {b}/{a}_{c}.log'.format(SOLVER_READY = settings['SOLVER_READY'], INITIAL_WEIGHTS = last_snapshot, c = name_for_stage, a = settings['experiment_name'], b = settings['LOGS_PATH']))

def removeExistingSnapshots(name_stage):
    snapshot_prefix_looked_for = '%s_%s' % (settings['experiment_name'], name_stage)
    available_snapshots = [x for x in os.listdir(settings['SNAPSHOT_PREFIX']) if 'caffemodel' in x and snapshot_prefix_looked_for in x]
    for snapshot_file in available_snapshots:
        os.remove(settings['SNAPSHOT_PREFIX'] + "/" + snapshot_file)
    available_solverstate = [x for x in os.listdir(settings['SNAPSHOT_PREFIX']) if 'solverstate' in x and snapshot_prefix_looked_for in x]
    for solverstate_file in available_solverstate:
        os.remove(settings['SNAPSHOT_PREFIX'] + "/" + solverstate_file)

def getLastAvailableSnapshot(name_previous_stage):
    snapshot_prefix_looked_for = '%s_%s' % (settings['experiment_name'], name_previous_stage)
    list_candidates = [(x.split("_")[-1],x) for x in os.listdir(settings['SNAPSHOT_PREFIX']) if 'caffemodel' in x and snapshot_prefix_looked_for in x]
    last_snapshot=settings['SNAPSHOT_PREFIX'] + "/" + sorted([(int(x.split(".")[0]), name) for x,name in list_candidates], key = lambda x: x[0], reverse = True)[0][1]
    return last_snapshot

if __name__ == '__main__':
    output_classes = _aux_getNumberOfCasses(settings['output_file_train'])
    initial_weights = settings['INITIAL_WEIGHTS']
    print("Training network with name " + settings['experiment_name'] + " which has " + str(output_classes) + ' classes')

    print("- Stage 1 with original weights: " + initial_weights)
    trainStage(initial_weights                      , output_classes, 1., 0., 250, '1st_stage')
    print("- Stage 2 starting with the resulting weights from 1st stage: " + getLastAvailableSnapshot('1st_stage'))
    trainStage(getLastAvailableSnapshot('1st_stage'), output_classes, 0.5, 0.5, 2500, '2nd_stage')

