import os
from PythonicNetwork import TestNetwork
from _stupid_tools_and_helpers_scripting import _aux_getNumberOfCasses, _aux_getSnapshotToBeused, _getPredefinedVariables_
import time
import numpy as np

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
        predictions.append(np.copy(net.getOutputData(imagenames)))

    if len(dataset) % batch_size != 0:
        print(" - last batch")
        first_index = batch_size*(len(dataset)/batch_size)
        imagenames = [dataset[first_index+j][0] for j in range(len(dataset) % batch_size)]
        labels += [map(int, list(set(dataset[first_index+j][1].split(" ")))) for j in range(len(dataset) % batch_size)]
        hashes += imagenames
        predictions.append(np.copy(net.getOutputData(imagenames)))

    return np.vstack(predictions)[:len(labels)], labels, hashes


def getAccuracy(predictions, labels):
    accs = 0.
    preds = predictions.argmax(axis=1)
    for i in range(predictions.shape[0]):
        if preds[i] in labels[i]:
            accs += 1
    return accs / predictions.shape[0]


if __name__ == '__main__':
    os.system('mkdir -p data/raw_results')

    #### VARIABLES

    ## These variables should be hardcoded
    CLASSIFIER_NAME = 'midtag'
    OUTPUT_CLASSES = _aux_getNumberOfCasses("experiments/chosing_one_tag/data/files/filtered_train.txt")
    model_file = _aux_getSnapshotToBeused(CLASSIFIER_NAME)
    print("Testing " + CLASSIFIER_NAME + " with " + str(OUTPUT_CLASSES) + " classes. Snapshot " + model_file)



    ## This loads output classes. It can be hardcoded
    vrs = _getPredefinedVariables_(CLASSIFIER_NAME)
    net = TestNetwork(OUTPUT_CLASSES, vrs['prototxt_base'], vrs['prototxt_ready'], model_file, vrs['batch_size'], vrs['imshape'], settings['map_template2file']['TEST'])

    t1 = time.time()
    predictions_val, labels_val, hashes_val = predictFromFile(net, "experiments/chosing_one_tag/data/files/filtered_val.txt")
    print(time.time()-t1)
    print("Final accuracy: " + str(getAccuracy(predictions_val, labels_val)))
    np.save('experiments/chosing_one_tag/data/raw_results/val_results.npy', predictions_val)
    np.save('experiments/chosing_one_tag/data/raw_results/val_hashes.npy', hashes_val)
    np.save('experiments/chosing_one_tag/data/raw_results/val_label.npy', labels_val)

    t1 = time.time()
    predictions_train, labels_train, hashes_train = predictFromFile(net, "experiments/chosing_one_tag/data/files/filtered_train.txt")
    print(time.time()-t1)
    print("Final accuracy: " + str(getAccuracy(predictions_train, labels_train)))
    np.save('experiments/chosing_one_tag/data/raw_results/train_results.npy', predictions_train)
    np.save('experiments/chosing_one_tag/data/raw_results/train_hashes.npy', hashes_train)
    np.save('experiments/chosing_one_tag/data/raw_results/train_label.npy', labels_train)

