## Take into account that if you want to use SoftmaxLabel you need one single label: <BATCH_SIZE, 1> but if you want the multilabel version you need <BATCH_SIZE, NUMBER_OF_LABELS>

## http://chrischoy.github.io/research/caffe-python-layer/
import caffe
import numpy as np

#### Multilabel-rara accuracy
# Bottoms 
# - predictions (after softmax)
# - gold labels (original ones) 
# Top
# - is a loss. The output must be size 1
# Out:
# - v1) Is the most probable predicted label in the possible ones?
#          #ss = []
           #for i in range(data[i_batch].shape[0]):
           #    if data[i_batch][i] == 1:
           #        ss.append(i)
           #print("-----------")
           #print(ss)
           #print(decided_label)
# - v2) If some of the K-firsts most probable ones between the possible labels?
#

class MonolabelAccuracyOneCorrectAfterSoftmax_top5(caffe.Layer):
    #This takes two inputs, a predicted probability distribution and the labels as a dense vector.
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        base_labels = bottom[1].data

        aind = bottom[0].data
        inds = np.array([sorted(range(len(aind[i])), key=lambda k: aind[i][k], reverse = True) for i in range(len(aind))])
        
        res = []
        for i in range(5):
            preds = inds[:,i]
            res.append(base_labels[range(base_labels.shape[0]), preds])
        result = np.max(res, axis=0) 

        top[0].data[...] = np.mean(result)

    def backward(self, top, propagate_down, bottom):
        pass


class MonolabelAccuracyOneCorrectAfterSoftmax(caffe.Layer):
    #This takes two inputs, a predicted probability distribution and the labels as a dense vector.
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        base_labels = bottom[1].data
        preds = np.argmax(bottom[0].data,axis=1)
        top[0].data[...] = np.mean(base_labels[range(base_labels.shape[0]), preds])

    def backward(self, top, propagate_down, bottom):
        pass








class MultilabelToMultilabelAttentionLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
    
        self.indexs = np.zeros([bottom[0].shape[0], bottom[0].shape[1]])
        for i_batch in range(self.indexs.shape[0]):
            for i in range(self.indexs.shape[1]):
                self.indexs[i_batch][i] = i
        self.new_labels = np.zeros([bottom[0].shape[0], bottom[0].shape[1]])

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        top[0].reshape(bottom[0].shape[0] * bottom[0].shape[1])

    def reshape(self, bottom, top):
        # We only reshape once because the input has constant shape
        pass
        # check input dimensions match

    def forward(self, bottom, top):
        inds_for_valid_labels = self.indexs[bottom[1].data.astype(bool)]
        probs_for_valid_labels = bottom[0].data[0,0,0,inds_for_valid_labels]
        probs_for_valid_labels = probs_for_valid_labels / probs_for_valid_labels.sum()
        decided_label = np.argmax(np.random.multinomial(1, probs_for_valid_labels, size=1)[0])
        self.new_labels *= 0
        self.new_labels[decided_label] = 1

        top[0].data[...] = self.new_labels

    def backward(self, top, propagate_down, bottom):
        pass


class MultilabelToMonolabelAttentionLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        
        self.indexs = np.zeros(bottom[0].shape[1]).astype(int)
        for i_batch in range(self.indexs.shape[0]):
            self.indexs[i_batch] = i_batch

        self.new_labels = np.zeros(bottom[0].shape[0]).astype(int)

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")

        top[0].reshape(bottom[0].shape[0])

    def reshape(self, bottom, top):
        # We only reshape once because the input has constant shape
        pass
        # check input dimensions match

    def forward(self, bottom, top):
        self.new_labels *= 0
        data = bottom[1].data
        for i_batch in range(bottom[0].shape[0]):
            inds_for_valid_labels = self.indexs[data[i_batch].astype(bool)]

            probs_for_valid_labels = bottom[0].data[i_batch, inds_for_valid_labels]
            probs_for_valid_labels = probs_for_valid_labels / probs_for_valid_labels.sum()
            # sometimes the sum is bigger than 1... we need to do something with this even if it is not nice
            total_prob = probs_for_valid_labels.sum()
            if total_prob >= 0.95:
                decided_label = inds_for_valid_labels[probs_for_valid_labels.argmax()]
            else:
                decided_label = inds_for_valid_labels[np.argmax(np.random.multinomial(1, probs_for_valid_labels, size=1)[0])]
            self.new_labels[i_batch] = decided_label

            #ss = []
            #for i in range(data[i_batch].shape[0]):
            #    if data[i_batch][i] == 1:
            #        ss.append(i)
            #print("-----------")
            #print(ss)
            #print(decided_label)

        top[0].data[...] = self.new_labels



    def backward(self, top, propagate_down, bottom):
        pass



class DummyAttentionMonolabelLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        top[0].reshape(bottom[0].shape)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        pass
