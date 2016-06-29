


from PIL import Image
from tools import SimpleTransformer
import scipy
import numpy as np
import caffe

imshape = (224,224)
prototxt_file = "data/base_network/my_network/ready_files/ready_network.prototxt"
model_file = "data/snapshots/snapshot_stage_1_iter_2500.caffemodel"
data_filename = "data/files/filtered_val.txt"
dataset = [line.rstrip('\n').split(",") for line in open(data_filename)]


st = SimpleTransformer()


 


net = caffe.Net(prototxt_file, model_file, caffe.TEST)
net.blobs['data'].reshape(1,3,imshape[0],imshape[1])

im = np.asarray(Image.open(dataset[0][0])) # load image
im = scipy.misc.imresize(im, imshape) # resize
im = st.preprocess(im)

net.blobs['data'].data[...] = im.reshape([1, im.shape[0], im.shape[1], im.shape[2]])
out = net.forward()
print(out['prob_out'])


