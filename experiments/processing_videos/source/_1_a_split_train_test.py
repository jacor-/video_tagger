###
#
# This script transforms the provided labels into categorical variables numbered between [0-NUM_CLASSES-1]. A file called classes.npy
# is saved in order to allow the translation of the final predicted labels into the original labels.
#
###


## You can load the LabelEncoder (to get the text instead of the label number):
#encoder = LabelEncoder()
#encoder.classes_ = numpy.load('data/files/classes.npy')

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from settings import settings

minimum_samples_per_tag = settings['minimum_samples_per_tag']




df = pd.read_csv(settings['dataset_filename'], sep = '#', header = None)

df[1] = df[1].map(lambda x: x[1:-1].replace("\"","").split(","))

labels = []
for val in df[1].values:
    labels += val

df2 = pd.DataFrame(columns = ['a'])
df2['a'] = labels
aux = df2.a.value_counts()
accepted_labels = aux[aux > minimum_samples_per_tag].index.values


le = LabelEncoder()
le.fit(accepted_labels)


df['labels'] = df[1].map(lambda x: le.transform([y for y in x if y in accepted_labels]))

should_be_empty = [x for x in list(set(np.hstack(df['labels'].values))) if x not in le.transform(accepted_labels)]
assert len(should_be_empty) == 0, 'Fuck shit!'

df = df[df['labels'].map(len) > 0]

df[[0, 'labels', 2]].to_csv(settings['processed_labels_csv'], index = False, header = False)


np.save(settings['processed_labels_2_original_label'], le.classes_)

