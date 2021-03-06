from time import time
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from settings import settings
import pandas as pd
'''
INSTRUCTIONS


1) You should copy all files from '/home/ubuntu/victor_tests/vilynx_bitbucket/vilynx-dl2/data/raw_results/.' to your 'path_source' folder
2) Your should copy the file '/home/ubuntu/victor_tests/vilynx_bitbucket/vilynx-dl2/data/files/classes.npy' into your 'path_source' folder

'''
summaries = [1, 2, 3]
frames = [0, 6, 12, 18]

def _loadNNData_():
    CLASSIFIER_NAME = settings['experiment_name']
    train_hashes = np.load('%s/data/raw_results/%s_train_hashes.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    train_labels = np.load('%s/data/raw_results/%s_train_label.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    train_results = np.load('%s/data/raw_results/%s_train_results.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    val_hashes = np.load('%s/data/raw_results/%s_val_hashes.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    val_labels = np.load('%s/data/raw_results/%s_val_label.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    val_results = np.load('%s/data/raw_results/%s_val_results.npy' % (settings['experiment_path'], CLASSIFIER_NAME))
    classes_en = np.load(settings['processed_labels_2_original_label'])
    return [train_hashes, train_labels, train_results, val_hashes, val_labels, val_results, classes_en]

def getResults():

    [train_hashes, train_labels, train_results, val_hashes, val_labels, val_results, classes_en] = _loadNNData_()
    corpus_train = {}
    corpus_val = {}

    for train_hash, train_label, train_result in zip(train_hashes, train_labels, train_results):
        train_hash = train_hash.split('/')[-1][:-4].split("_")
        hash_, summary, frame = train_hash[0], int(train_hash[1]), int(train_hash[2])
        if frame in frames and summary in summaries:
            if hash_ not in corpus_train:
                corpus_train[hash_] = {'label': train_label, 'X':[train_result]}
            else:
                corpus_train[hash_]['X'].append(train_result)

    for val_hash, val_label, val_result in zip(val_hashes, val_labels, val_results):
        val_hash = val_hash.split('/')[-1][:-4].split("_")
        hash_, summary, frame = val_hash[0], int(val_hash[1]), int(val_hash[2])
        if frame in frames and summary in summaries:
	        if hash_ not in corpus_val:
	            corpus_val[hash_] = {'label': val_label, 'X':[val_result]}
	        else:
	            corpus_val[hash_]['X'].append(val_result)

    return [corpus_train, corpus_val]


def getDatasets_aggrsum(corpus_train, corpus_val):
    X_train = np.array([np.sum(corpus_train[key]['X'], axis=0) for key in corpus_train.keys()])
    X_test  = np.array([np.sum(corpus_val[key]['X'], axis=0) for key in corpus_val.keys()])


    y_train = [corpus_train[key]['label'] for key in corpus_train.keys()]
    y_val   = [corpus_val[key]['label'] for key in corpus_val.keys()]

    N_classes = np.max([np.max(np.max(y_val))+1,np.max(np.max(y_train))+1])

    out_space_val = np.zeros([len(y_val), N_classes])
    for i in range(len(y_val)):
        out_space_val[i][np.array(y_val[i])] = 1
	out_space_train = np.zeros([len(y_train), N_classes])
    for i in range(len(y_train)):
        out_space_train[i][np.array(y_train[i])] = 1


    return X_train, X_test, out_space_train, out_space_val

def getDatasets_baseline(corpus_train, corpus_val):
    ## We compute the baseline by simply getting the maximum for each one of the outputs. So no classifier is used here!
    X_train = [np.argmax(corpus_train[key]['X'], axis=1) for key in corpus_train.keys()]
    X_test  = [np.argmax(corpus_val[key]['X']  , axis=1) for key in corpus_val.keys()  ]

    y_train = [corpus_train[key]['label'] for key in corpus_train.keys()]
    y_val   = [corpus_val[key]['label'] for key in corpus_val.keys()]

    N_classes = np.max([np.max(np.max(y_val))+1,np.max(np.max(y_train))+1])

    # We fill the validation space by using the known outputs
    out_space_val = np.zeros([len(y_val), N_classes])
    for i in range(len(y_val)):
        out_space_val[i][np.array(y_val[i])] = 1
    out_space_train = np.zeros([len(y_train), N_classes])
    for i in range(len(y_train)):
        out_space_train[i][np.array(y_train[i])] = 1

    # We fill the the predicted space by taking the maximum in Train and Test
    out_space_X_val = np.zeros([len(y_val), N_classes])
    for i in range(len(X_test)):
        out_space_X_val[i][np.array(X_test[i])] = 1
    out_space_X_train = np.zeros([len(y_train), N_classes])
    for i in range(len(X_train)):
        out_space_X_train[i][np.array(X_train[i])] = 1

    return out_space_X_train, out_space_X_val, out_space_train, out_space_val


def performMetaClassification(X_train, X_test, y_train, y_test, type_="OutputCodeClassifier"):
    print(type_)
    if type_ == "RandomForestClassifier":
        model = RandomForestClassifier(n_jobs = -1, n_estimators = 250)

    elif type_ == "KNeighborsClassifier":
        model = KNeighborsClassifier(10)

    elif type_ == "OutputCodeClassifier":
        model = OutputCodeClassifier(LinearSVC(random_state=0))

    elif type_ == "OneVsRestClassifier":
        model = OneVsRestClassifier(LinearSVC(random_state=0))

    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    return preds_test


metric_functions = {
   'strict_accuracy' : lambda pred, ref: float(((pred - ref).sum(axis=1) == 0).sum()) / ref.shape[0],
   'at_least_one_no_mistake' : lambda pred, ref: (((pred * ref).sum(axis=1) > 0) & ((ref - pred == -1).sum(axis=1) ==0)).mean()
}


if __name__ == "__main__":
    [corpus_train, corpus_val] = getResults()

    df_results = pd.DataFrame(columns = ['name','split','metric','value'])


    # We collect results based on meta classifiers
    [X_train, X_test, y_train, y_test] = getDatasets_aggrsum(corpus_train, corpus_val)
    print(X_train.shape)
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test
    for classifier_name in ['RandomForestClassifier']:
        t1 = time.time()
        preds_test = performMetaClassification(X_train, X_test, y_train, y_test, classifier_name)
        print(time.time()-t1)
        for metric_name in metric_functions:
            df_results.loc[df_results.shape[0]+1] = [classifier_name,'Test',metric_name,metric_functions[metric_name](preds_test, y_test)]
        #for metric_name in metric_functions:
        #    df_results.loc[df_results.shape[0]+1] = [classifier_name,'Train',metric_name,metric_functions[metric_name](preds_train, y_train)]


    # We collect baseline reuslts
    [X_train, X_test, y_train, y_test] = getDatasets_baseline(corpus_train, corpus_val)
    for metric_name in metric_functions:
        df_results.loc[df_results.shape[0]+1] = ['Baseline','Test',metric_name,metric_functions[metric_name](X_test, y_test)]
    #for metric_name in metric_functions:
    #    df_results.loc[df_results.shape[0]+1] = ['Baseline','Train',metric_name,metric_functions[metric_name](X_train, y_train)]


    # Lets see our results!
    df_results.to_csv('%s/data/files/%s_metrics.csv'  % (settings['experiment_path'], settings['experiment_name']))


    print(df_results)











