


from PIL import Image
from tools import SimpleTransformer
import scipy
import numpy as np
import caffe
import template_tools
from template_tools.template_manager1 import PrototxtTemplate
import subprocess
import os
import sys




class TestNetwork(object):

    def __init__(self, OUTPUTNEURONS, prototxt_base, prototxt_ready, model_file, max_batch_size, imshape):

        self.OUTPUTNEURONS = OUTPUTNEURONS
        self.prototxt_ready = prototxt_ready
        self.max_batch_size = max_batch_size
        self.data_container = np.zeros([max_batch_size,3,imshape[0],imshape[1]])


        self._prepareDeployPrototxts_(prototxt_base)
        net = caffe.Net(prototxt_ready, model_file, caffe.TEST)
        net.blobs['data'].reshape(max_batch_size,3,imshape[0],imshape[1])
        self.net = net

    def _prepareDeployPrototxts_(self, prototxt_base):
        variables_to_replace = {
            'LRMULTBASENET' : '0',
            'DEMULTBASENET' : '0',
            'LRMULTLASTLAYER' : '1',
            'DEMULTLASTLAYER' : '2',
            'OUTPUTNEURONS' : str(self.OUTPUTNEURONS),
        }

        map_template2file = {
            'inputprototxt' :                   './base_network/my_network/base_files/input_layers_base/deploy_input_layers_base.prototxt',
            'evaltrainstage' :                  './base_network/my_network/base_files/output_layers_templates/empty_template.prototxt',
            'crossentropylossintermediate' :    './base_network/my_network/base_files/output_layers_templates/empty_template.prototxt'
        }

        new_prototxt = PrototxtTemplate(prototxt_base, map_template2file)
        new_prototxt.saveOutputPrototxt(self.prototxt_ready, variables_to_replace)


    def _loadData_(self, imagenames):
        self.data_container *= 0
        st = SimpleTransformer()

        for i in range(len(imagenames)):
            ims = np.asarray(Image.open(imagenames[i])) # load image
            ims = scipy.misc.imresize(ims, imshape) # resize
            ims = st.preprocess(ims)
            self.data_container[i] = ims
        return self.data_container

    def getOutputData(self, imagenames):
        data = self._loadData_(imagenames)
        self.net.blobs['data'].data[...] = data #data.reshape([data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
        out = self.net.forward()
        return out['probsout']

def predictFromFile(net, input_data_file):
    data_filename = input_data_file
    dataset = [line.rstrip('\n').split(",") for line in open(data_filename)]

    predictions = []
    labels = []

    print("Predicting %d samples from file %s" % (len(dataset),input_data_file))
    for i in range(len(dataset)/10):
        print(" - batch %d" %i)
        imagenames = [dataset[i*10+j][0] for j in range(10)]
        labels.append([map(int, list(set(dataset[i*10+j][1].split(" ")))) for j in range(10)])

        predictions.append(net.getOutputData(imagenames))

    if len(dataset) % 10 != 0:
        print(" - last batch")
        first_index = 10*(len(dataset)/10)
        imagenames = [dataset[first_index+j][0] for j in range(len(dataset) % 10)]
        labels.append([map(int, list(set(dataset[first_index+j][1].split(" ")))) for j in range(len(dataset) % 10)])

        predictions.append(net.getOutputData(imagenames))

    return predict, labels

def getPredefinedVariables():
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

if __name__ == '__main__':
    #### VARIABLES

    ## These variables should be hardcoded
    CLASSIFIER_NAME = 'midtag'
    OUTPUT_CLASSES = _aux_getNumberOfCasses("./data/files/filtered_train.txt")
    print("Testing " + CLASSIFIER_NAME + " with " + str(OUTPUT_CLASSES) + " classes. Snapshot: " + model_file)


    ## This loads output classes. It can be hardcoded
    vrs = getPredefinedVariables()
    net = TestNetwork(OUTPUT_CLASSES, vrs['prototxt_base'], vrs['prototxt_ready'], vrs['model_file'], vrs['batch_size'], vrs['imshape'])

    predictFromFile(net, "data/files/filtered_val.txt")

