
## You can load the LabelEncoder (to get the text instead of the label number):
#encoder = LabelEncoder()
#encoder.classes_ = numpy.load('data/files/classes.npy')


minimum_samples_per_tag = 30



from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv('base_network/VictorData/dataset_30c.csv', sep = '#', header = None)

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

df[[0, 'labels', 2]].to_csv('data/files/prepared_dataset.csv', index = False, header = False)


np.save('data/files/classes.npy', le.classes_)

#encoder = LabelEncoder()
#encoder.classes_ = numpy.load('data/files/classes.npy')

