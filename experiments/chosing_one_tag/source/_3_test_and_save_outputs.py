import os
from PythonNetwork.PythonicNetwork import TestNetwork
from _stupid_tools_and_helpers_scripting import _aux_getNumberOfCasses, _aux_getSnapshotToBeused, _getPredefinedVariables_
import time
import numpy as np
from settings import settings

def predictFromFile(net, input_data_file):
    data_filename = input_data_file
    dataset = [line.rstrip('\n').split(",") for line in open(data_filename)]

    predictions = []
    labels = []
    hashes = []
    print("Predicting %d samples from file %s" % (len(dataset),data_filename))
    batch_size = net.max_batch_size
    for i in range(len(dataset)/batch_size):
        print(" - batch %d out of %d" % (i, len(dataset)/batch_size))
        imagenames = [dataset[i*batch_size+j][0] for j in range(batch_size)]
        labels += [map(int, list(set(dataset[i*batch_size+j][1].split(" ")))) for j in range(batch_size)]
        hashes += imagenames
        predictions.append(np.copy(net.getOutputData(imagenames, ['probs'])[0]))

    if len(dataset) % batch_size != 0:
        print(" - last batch")
        first_index = batch_size*(len(dataset)/batch_size)
        imagenames = [dataset[first_index+j][0] for j in range(len(dataset) % batch_size)]
        labels += [map(int, list(set(dataset[first_index+j][1].split(" ")))) for j in range(len(dataset) % batch_size)]
        hashes += imagenames
        predictions.append(np.copy(net.getOutputData(imagenames, ['probs'])[0]))

    return np.vstack(predictions)[:len(labels)], labels, hashes


def getAccuracy(predictions, labels):
    accs = 0.
    preds = predictions.argmax(axis=1)
    for i in range(predictions.shape[0]):
        if preds[i] in labels[i]:
            accs += 1
    return accs / predictions.shape[0]


def getLastAvailableSnapshot(stage_name = '2nd_stage'):
    snapshot_prefix_looked_for = '%s_%s' % (settings['experiment_name'], stage_name)
    list_candidates = [(x.split("_")[-1],x) for x in os.listdir(settings['SNAPSHOT_PREFIX']) if 'caffemodel' in x and snapshot_prefix_looked_for in x]
    last_snapshot=settings['SNAPSHOT_PREFIX'] + "/" + sorted([(int(x.split(".")[0]), name) for x,name in list_candidates], key = lambda x: x[0], reverse = True)[0][1]
    return last_snapshot

if __name__ == '__main__':
    os.system('mkdir -p data/raw_results')

    #### VARIABLES

    ## These variables should be hardcoded
    CLASSIFIER_NAME = settings['experiment_name']
    OUTPUT_CLASSES = _aux_getNumberOfCasses(settings['output_file_train'])

    #model_file = getLastAvailableSnapshot()
    model_file = 'experiments/chosing_one_tag/data/snapshots/exploring_multiframe_2nd_stage_iter_3750.caffemodel'

    print("Testing " + CLASSIFIER_NAME + " with " + str(OUTPUT_CLASSES) + " classes. Snapshot " + model_file)



    ## This loads output classes. It can be hardcoded
    vrs = _getPredefinedVariables_(CLASSIFIER_NAME)
    net = TestNetwork(OUTPUT_CLASSES, vrs['prototxt_base'], vrs['prototxt_ready'], model_file, vrs['batch_size'], vrs['imshape'], settings['map_template2file']['TEST'])


    t1 = time.time()
    predictions_val, labels_val, hashes_val = predictFromFile(net, "experiments/chosing_one_tag/data/files/val.txt")
    print(time.time()-t1)
    print("Final accuracy: " + str(getAccuracy(predictions_val, labels_val)))
    np.save('%s/data/raw_results/%s_val_results.npy' % (settings['experiment_path'], CLASSIFIER_NAME), predictions_val)
    np.save('%s/data/raw_results/%s_val_hashes.npy' % (settings['experiment_path'], CLASSIFIER_NAME), hashes_val)
    np.save('%s/data/raw_results/%s_val_label.npy' % (settings['experiment_path'], CLASSIFIER_NAME), labels_val)

    t1 = time.time()
    predictions_train, labels_train, hashes_train = predictFromFile(net, "experiments/chosing_one_tag/data/files/train.txt")
    print(time.time()-t1)
    print("Final accuracy: " + str(getAccuracy(predictions_train, labels_train)))
    np.save('%s/data/raw_results/%s_train_results.npy' % (settings['experiment_path'], CLASSIFIER_NAME), predictions_train)
    np.save('%s/data/raw_results/%s_train_hashes.npy' % (settings['experiment_path'], CLASSIFIER_NAME), hashes_train)
    np.save('%s/data/raw_results/%s_train_label.npy' % (settings['experiment_path'], CLASSIFIER_NAME), labels_train)

