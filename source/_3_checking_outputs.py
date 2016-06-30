


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

imshape = (224,224)
prototxt_base='./base_network/my_network/base_files/googlenetbase.prototxt'
prototxt_ready='./data/base_network/my_network/ready_files/ready_network_deploy.prototxt'
#snapshot_prefix_looked_for = 'snapshot_stage_1'
#model_file='data/snapshots/' + sorted([(int(x.split(".")[0]), name) for x,name in [(x.split("_")[-1],x) for x in os.listdir('data/snapshots') if 'caffemodel' in x and snapshot_prefix_looked_for in x]], key = lambda x: x[0], reverse = True)[0][1]
model_file='/home/ubuntu/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

data_filename = "data/files/filtered_val.txt"
dataset = [line.rstrip('\n').split(",") for line in open(data_filename)]


TRAIN_FILENAME="./data/files/filtered_train.txt";
command = "cat {train_filename}  | cut -d ',' -f2 | tr ' ' '\n' | sort | uniq | wc -l".format(train_filename = TRAIN_FILENAME)
OUTPUT_CLASSES= int(subprocess.check_output(command, shell = True))


variables_to_replace = {
    'LRMULTBASENET' : '0',
    'DEMULTBASENET' : '0',
    'LRMULTLASTLAYER' : '1',
    'DEMULTLASTLAYER' : '2',
    'OUTPUTNEURONS' : str(OUTPUT_CLASSES),
}

map_template2file = {
    'inputprototxt' :                   './base_network/my_network/base_files/input_layers_base/deploy_input_layers_base.prototxt',
    'evaltrainstage' :                  './base_network/my_network/base_files/output_layers_templates/empty_template.prototxt',
    'crossentropylossintermediate' :    './base_network/my_network/base_files/output_layers_templates/empty_template.prototxt'
}

new_prototxt = PrototxtTemplate(prototxt_base, map_template2file)
new_prototxt.saveOutputPrototxt(prototxt_ready, variables_to_replace)






def loadData(imagename, st):
    ims = np.asarray(Image.open(imagename)) # load image
    ims = scipy.misc.imresize(im, imshape) # resize
    ims = st.preprocess(im)
    return ims

def getOutputData(net, image):
    net.blobs['data'].data[...] = im.reshape([1, im.shape[0], im.shape[1], im.shape[2]])
    out = net.forward()
    return out['probsout']



def initNetwork(prototxt, model_file):
    net = caffe.Net(prototxt, model_file, caffe.TEST)
    net.blobs['data'].reshape(1,3,imshape[0],imshape[1])
    return net


st = SimpleTransformer()

## Load networks (all those networks you will need)
net = initNetwork(prototxt_ready, model_file)

## Load images and predict. It does not accept batches atm. It will, though
outputs = []
for i in range(10):
    im = loadData(dataset[0][i], st)
    out = getOutputData(net, im)
    outputs.append(out)
