
from PythonicNetwork import TestNetwork
from _stupid_tools_and_helpers_scripting import _aux_getNumberOfCasses, _aux_getSnapshotToBeused, _getPredefinedVariables_
import time
import numpy as np

def predictFromFile(net, input_data_file):
    data_filename = input_data_file
    dataset = [line.rstrip('\n').split(",") for line in open(data_filename)]

    predictions = []
    labels = []

    print("Predicting %d samples from file %s" % (len(dataset),data_filename))
    batch_size = net.max_batch_size
    for i in range(len(dataset)/batch_size):
        print(" - batch %d out of %d" % (i, len(dataset)/batch_size))
        imagenames = [dataset[i*batch_size+j][0] for j in range(batch_size)]
        labels += [map(int, list(set(dataset[i*batch_size+j][1].split(" ")))) for j in range(batch_size)]

        predictions.append(np.copy(net.getOutputData(imagenames)))

    if len(dataset) % batch_size != 0:
        print(" - last batch")
        first_index = batch_size*(len(dataset)/batch_size)
        imagenames = [dataset[first_index+j][0] for j in range(len(dataset) % batch_size)]
        labels += [map(int, list(set(dataset[first_index+j][1].split(" ")))) for j in range(len(dataset) % batch_size)]

        predictions.append(np.copy(net.getOutputData(imagenames)))

    return np.vstack(predictions)[:len(labels)], labels


def getAccuracy(predictions, labels):
    accs = 0.
    preds = predictions.argmax(axis=1)
    for i in range(predictions.shape[0]):
        if preds[i] in labels[i]:
            accs += 1
    return accs / predictions.shape[0]


if __name__ == '__main__':
    #### VARIABLES

    ## These variables should be hardcoded
    CLASSIFIER_NAME = 'midtag'
    OUTPUT_CLASSES = _aux_getNumberOfCasses("./data/files/filtered_train.txt")
    model_file = _aux_getSnapshotToBeused(CLASSIFIER_NAME)


    ## This loads output classes. It can be hardcoded
    vrs = _getPredefinedVariables_(CLASSIFIER_NAME)
    net = TestNetwork(OUTPUT_CLASSES, vrs['prototxt_base'], vrs['prototxt_ready'], model_file, vrs['batch_size'], vrs['imshape'])

    print("Testing " + CLASSIFIER_NAME + " with " + str(OUTPUT_CLASSES) + " classes. Snapshot " + model_file)

    t1 = time.time()
    predictions, labels = predictFromFile(net, "data/files/filtered_val.txt")
    print(time.time()-t1)

    print("Final accuracy: " + str(getAccuracy(predictions, labels)))
