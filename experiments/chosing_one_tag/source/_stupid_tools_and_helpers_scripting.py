import subprocess
import os


def _getPredefinedVariables_(CLASSIFIER_NAME):
    return {
        'batch_size' : 500,
        'imshape' : (224,224),
        'prototxt_base' : './base_network/my_network/base_files/googlenetbase.prototxt',
        'prototxt_ready' : './data/base_network/my_network/ready_files/%s_ready_network_deploy.prototxt' % CLASSIFIER_NAME
    }

def _aux_getNumberOfCasses(filename):
    TRAIN_FILENAME=filename
    command = "cat {train_filename}  | cut -d ',' -f2 | tr ' ' '\n' | sort | uniq | wc -l".format(train_filename = TRAIN_FILENAME)
    OUTPUT_CLASSES=int(subprocess.check_output(command, shell = True))
    return OUTPUT_CLASSES

def _aux_getSnapshotToBeused(classifier_name):
    snapshot_prefix_looked_for = '%s_snapshot_stage_2' % classifier_name
    model_file='data/snapshots/' + sorted([(int(x.split(".")[0]), name) for x,name in [(x.split("_")[-1],x) for x in os.listdir('data/snapshots') if 'caffemodel' in x and snapshot_prefix_looked_for in x]], key = lambda x: x[0], reverse = True)[0][1]
    return model_file

