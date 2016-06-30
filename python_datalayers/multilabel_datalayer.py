import json, time, pickle, scipy.misc, skimage.io, caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


##Params is a dict:
#params = {
#    'batch_size':50,
#    'split':'train', #('train','test','val')
#    'data_filename':?
#    'im_shape':
#}

class VilynxDatabaseSync(caffe.Layer):
    """
    This is a simple syncronous datalayer for training a multilabel model on PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)


        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        assert 'data_filename' in params.keys(), 'Params must include data_filename.'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
        assert 'N_labels' in params.keys(), 'Params must include the total number of labels considered'
        # store input as class variables
        self.N_labels = params['N_labels']
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.data_filename = params['data_filename']
        # TODO: read train, test, val files...
        #self.indexlist = [line.rstrip('\n') for line in open(osp.join(self.data_filename, 'ImageSets/Main', params['split'] + '.txt'))] #get list of image indexes.
        ## The expected share of my data is: "imagename,1 3 4 5"
        self.dataset = [line.rstrip('\n').split(",") for line in open(self.data_filename)]

        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, self.N_labels)

        print "VilynxDatabaseSync initialized for split: {}, with bs:{}, im_shape:{}, and {} images.".format(params['split'], params['batch_size'], params['im_shape'], len(self.dataset))


    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.dataset):
                self._cur = 0
                shuffle(self.dataset)

            # Load an image
            
            index = self.dataset[self._cur] # Get the image index

            #TODO: load images path...
            im = np.asarray(Image.open(index[0])) # load image
            im = scipy.misc.imresize(im, self.im_shape) # resize

            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]

            # Load and prepare ground truth
            multilabel = np.zeros(self.N_labels).astype(np.float32)
            for label in map(int, index[1].split(" ")):
                # in the multilabel problem we don't care how MANY instances there are of each class. Only if they are present.
                multilabel[label] = 1 # The "-1" is b/c we are not interested in the background class.

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = self.transformer.preprocess(im)
            top[1].data[itt, ...] = multilabel
            self._cur += 1

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass




class VilynxDatabaseAsync(caffe.Layer):
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
        assert 'data_filename' in params.keys(), 'Params must include data_filename.'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
        assert 'N_labels' in params.keys(), 'Params must include the total number of labels considered'
        # store input as class variables
        self.N_labels = params['N_labels']
        self.batch_size = params['batch_size'] # we need to store this as a local variable.

        # === We are going to do the actual data processing in a seperate, helperclass, called BatchAdvancer. So let's forward the parame to that class ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = BatchAdvancer(self.thread_result, params)
        self.dispatch_worker() # Let it start fetching data right away.

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, self.N_labels)

        print "VilynxDatabaseAsync initialized for split: {}, with bs:{}, im_shape:{}.".format(params['split'], params['batch_size'], params['im_shape'])



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
        self.data_filename = params['data_filename']
        self.im_shape = params['im_shape']
        self.N_labels = params['N_labels']


        self.dataset = [line.rstrip('\n').split(",") for line in open(self.data_filename)]

        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        print "BatchAdvancer initialized with {} images".format(len(self.dataset))

    def __call__(self):
        """
        This does the same stuff as the forward layer of the synchronous layer. Exept that we store the data and labels in the result dictionary (as lists of length batchsize).
        """
        self.result['data'] = []
        self.result['label'] = []
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.dataset):
                self._cur = 0
                shuffle(self.dataset)

            # Load an image
            index = self.dataset[self._cur] # Get the image index


            #im = np.asarray(Image.open(osp.join(self.data_filename, 'JPEGImages', index + '.jpg'))) # load image
            #print('Loading image ' + str(index[0]))
            im = np.asarray(Image.open(index[0])) # load image
            im = scipy.misc.imresize(im, self.im_shape) # resize
            #print('oaded image ' + str(index[0]))
            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]



            # Load and prepare ground truth
            multilabel = np.zeros(self.N_labels).astype(np.float32)
            for label in map(int, index[1].split(" ")):
                # in the multilabel problem we don't care how MANY instances there are of each class. Only if they are present.
                multilabel[label] = 1 # The "-1" is b/c we are not interested in the background class.

            # Store in a result list.
            self.result['data'].append(self.transformer.preprocess(im))
            self.result['label'].append(multilabel)
            self._cur += 1

