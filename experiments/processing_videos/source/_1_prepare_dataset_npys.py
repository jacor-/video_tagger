from settings import settings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

print(" - Files will be saved here: " + settings['path_for_files'])
minimum_samples_per_tag = 30

df = pd.read_csv(settings['dataset_filename'], sep = ';', header = None)
print(df.columns)
print(df.head())
df.columns = ['hash','summary','position','mid','vid']
# Sanity check 1
frames_per_video = df.groupby(['vid']).count()['hash'].unique()
assert len(frames_per_video) == 1, 'The number of frames per video is not constant'

vid_tag = df.groupby('vid').first()[['mid']].reset_index()
vid_tag['mid'] = vid_tag['mid'].map(lambda x: x[1:-1].replace("\"","").split(","))
labels = np.hstack(vid_tag['mid'])

df2 = pd.DataFrame(columns = ['a'])
df2['a'] = labels
aux = df2.a.value_counts()
accepted_labels = aux[aux > settings['minimum_samples_per_tag']].index.values
print(" - The dataset had %d unique labels. The accepted ones where %d" % (len(list(set(labels))), len(accepted_labels)))

le = LabelEncoder()
le.fit(accepted_labels)

vid_tag['labels'] = vid_tag['mid'].map(lambda x: le.transform([y for y in x if y in accepted_labels]))
# Sanity check 2
should_be_empty = [x for x in list(set(np.hstack(vid_tag['labels'].values))) if x not in le.transform(accepted_labels)]
assert len(should_be_empty) == 0, 'Fuck shit!'

vid_tag = vid_tag[vid_tag['labels'].map(len) > 0]
frames_per_video_sorted = df[df.vid.isin(vid_tag.vid.values)].groupby('vid')['hash'].apply(lambda x: sorted(list(x), key = lambda x: int(x.split("_")[1])*100 + int(x.split("_")[2]))).reset_index()



df_dataset = pd.merge(vid_tag, frames_per_video_sorted, on = 'vid').set_index('vid')


dataset = {}
for vid in df_dataset.index.values:
	now_data = df_dataset.ix[vid]
	dataset[str(vid)] = {'labels': now_data['labels'], 'images': now_data['hash']}

np.save(settings['path_for_files'] + "/" + settings['dict_dataset'], dataset)
np.save(settings['path_for_files'] + "/" + settings['processed_labels_2_original_label'], le.classes_)



# Split train and test sets
available_vids = np.array(list(dataset.keys()))
indexs = np.arange(len(available_vids))
np.random.shuffle(indexs)
samples_in_train = int(np.floor(settings['train_size']*len(available_vids)))
train_samples = available_vids[indexs[:samples_in_train]]
test_samples = available_vids[indexs[samples_in_train:]]

f = open(settings['path_for_files'] + "/" + settings['output_file_train'], 'w')
for sample in train_samples:
	f.write(str(sample) + "\n")
f.close()
f = open(settings['path_for_files'] + "/" + settings['output_file_test'], 'w')
for sample in test_samples:
	f.write(str(sample) + "\n")
f.close()


