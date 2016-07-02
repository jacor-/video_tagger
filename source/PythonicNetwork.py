


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
import time



class TestNetwork(object):

    def __init__(self, OUTPUTNEURONS, prototxt_base, prototxt_ready, model_file, max_batch_size, imshape):

        self.OUTPUTNEURONS = OUTPUTNEURONS
        self.prototxt_ready = prototxt_ready
        self.max_batch_size = max_batch_size
        self.data_container = np.zeros([max_batch_size,3,imshape[0],imshape[1]])
        self.imshape = imshape

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
            ims = scipy.misc.imresize(ims, self.imshape) # resize
            ims = st.preprocess(ims)
            self.data_container[i] = ims
        return self.data_container

    def getOutputData(self, imagenames):
        data = self._loadData_(imagenames)
        self.net.blobs['data'].data[...] = data #data.reshape([data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
        out = self.net.forward()
        return out['probsout']

