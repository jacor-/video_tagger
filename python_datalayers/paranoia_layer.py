## Take into account that if you want to use SoftmaxLabel you need one single label: <BATCH_SIZE, 1> but if you want the multilabel version you need <BATCH_SIZE, NUMBER_OF_LABELS>

## http://chrischoy.github.io/research/caffe-python-layer/
import caffe
import numpy as np



import json, time, pickle, scipy.misc, skimage.io, caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer



class SmoothMaxVideoLayer(caffe.Layer):

    def setup(self, bottom, top):
        # We only have an input here: the result of processing each frame independently
        if len(bottom) != 1:
            raise Exception("We expect only one input.")

        params = eval(self.param_str)
        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'frames_per_video' in params.keys(), 'Params must include frames_per_video.'

        self.batch_size = int(params['batch_size'])
        self.frames_per_video = int(params['frames_per_video'])
        self.videos_per_batch = self.batch_size / self.frames_per_video

        print("Input shape: ")
        print(bottom[0].shape)

        #We will use e_xi as an auxiliar space. The mask will be used to perform vectorized operation on the output.
        self.e_xi = np.zeros(bottom[0].shape)
        self.mask = np.zeros((self.videos_per_batch, self.batch_size))
        for ivid in range(self.videos_per_batch):
            for iframe in range(self.frames_per_video):
                self.mask[ivid][ivid*self.frames_per_video+iframe] = 1

        top[0].reshape(self.videos_per_batch, bottom[0].shape[1])

    def reshape(self, bottom, top):
        # We only reshape once because the input has constant shape
        pass
        # check input dimensions match

    def forward(self, bottom, top):
        # assumes alpha = 1
        self.e_xi = np.exp(bottom[0].data)

        numerator = np.dot(self.mask, (bottom[0].data * self.e_xi))
        denominator = np.dot(self.mask, self.e_xi)
        output = numerator / denominator
        ### COMPUTE THE OUTPUT

        top[0].data[...] = output


        dif_right = (1 + bottom[0] - np.dot(self.mask.T, output))
        dif_left = self.e_xi / np.dot(self.mask.T, denominator)## This one should be doable using less memory
        self.diff[...] = dif_right * dif_left

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        bottom[i].diff[...] = self.diff






class VilynxDatabaseVideosAsync(caffe.Layer):
    """
    This is a simple asyncronous datalayer for training a multilabel model on PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        #print(self.param_str)
        params = eval(self.param_str)

        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        assert 'dataset_file' in params.keys(), 'Params must include data_filename.'
        assert 'samples_file' in params.keys(), 'Params must include data_filename.'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
        assert 'N_labels' in params.keys(), 'Params must include the total number of considered labels'
        assert 'image_path' in params.keys(), 'Params must include the image path'
        assert 'frames_per_video' in params.keys(), 'Params must include frames_per_video.'

        self.batch_size = params['batch_size']
        self.frames_per_video = params['frames_per_video']
        self.videos_per_batch = self.batch_size / self.frames_per_video

        # store input as class variables
        self.N_labels = params['N_labels']
        self.batch_size = params['batch_size'] # we need to store this as a local variable.

        # === We are going to do the actual data processing in a seperate, helperclass, called BatchAdvancer. So let's forward the parame to that class ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = BatchAdvancer(self.thread_result, params)
        self.dispatch_worker() # Let it start fetching data right away.

        # """ We check that the number of frames fits in the batch
        assert np.mod(self.batch_size, self.frames_per_video) == 0, "Batch size must be divisible by the number of frames per video"

        # === reshape tops ===
        # --> data will have size BATCH. It will be then projected into VIDEOS_PER_BATCH space in the last layer (SmoothMax)
        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.videos_per_batch, self.N_labels)

        print "VilynxDatabaseAsync initialized for split: {}, with batch size:{}, im_shape:{}.".format(params['split'], params['batch_size'], params['im_shape'])

    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass

    def forward(self, bottom, top):
        """ this is the forward pass, where we load the data into the blobs. Since we run the BatchAdvance asynchronously, we just wait for it, and then copy """

        if self.thread is not None:
            self.join_worker() # wait until it is done.

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                aux = self.thread_result
                #print(name)
                print("---")
                print(len(aux[name]))
                print(aux[name][0])
                print(aux[name][1])
                print(aux[name][2])
                print(aux[name][3])
                top[top_index].data[i, ...] = aux[name][i] #Copy the already-prepared data to caffe.

        self.dispatch_worker() # let's go again while the GPU process this batch.

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class BatchAdvancer():
    """
    This is the class that is run asynchronously and actually does the work.
    """
    def __init__(self, result, params):
        self.result = result

        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.im_shape = params['im_shape']
        self.N_labels = params['N_labels']
        self.image_path = params['image_path']


        self.frames_per_video = params['frames_per_video']
        self.videos_per_batch = self.batch_size / self.frames_per_video

        self.dataset = np.load(params['dataset_file']).item()
        self.video_list = [line[:-1] for line in open(params['samples_file']).readlines() if len(line) > 1]


        # store input as class variables
        self.N_labels = params['N_labels']
        self.batch_size = params['batch_size'] # we need to store this as a local variable.


        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        print("BatchAdvancer initialized with %d images".format(len(self.dataset.keys())))

    def __call__(self):
        """
        This does the same stuff as the forward layer of the synchronous layer. Exept that we store the data and labels in the result dictionary (as lists of length batchsize).
        """
        self.result['data'] = []
        self.result['label'] = []
        for itt in range(self.videos_per_batch):

            # Did we finish an epoch?
            if self._cur == len(self.video_list):
                self._cur = 0
                shuffle(self.video_list)

            # Load an image
            video_data = self.dataset[self.video_list[self._cur]] # Get the image index


            # Load and prepare ground truth
            labels = video_data['labels']
            multilabel = np.zeros(self.N_labels).astype(np.float32)
            for label in labels:
                # in the multilabel problem we don't care how MANY instances there are of each class. Only if they are present.
                multilabel[label] = 1 # The "-1" is b/c we are not interested in the background class.
            
            for i_frame in range(self.frames_per_video):
                imagepath = self.image_path + "/" + video_data['images'][i_frame]

                #im = np.asarray(Image.open(osp.join(self.data_filename, 'JPEGImages', index + '.jpg'))) # load image
                #print('Loading image ' + str(index[0]))
                im = np.asarray(Image.open(imagepath)) # load image
                im = scipy.misc.imresize(im, self.im_shape) # resize
                #print('oaded image ' + str(index[0]))
                # do a simple horizontal flip as data augmentation
                flip = np.random.choice(2)*2-1
                im = im[:, ::flip, :]
                self.result['data'].append(self.transformer.preprocess(im))

                # Store in a result list.
                self.result['label'].append(multilabel)

            
            self._cur += 1


















