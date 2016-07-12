####
## We split the samples prepared by the previous script in train and validation and store the resulting datasets in two different files (caffe style)
##
## Please, notice that the last step is to check that all the images we write in the training files actually exists in the image folder.
####
import os
import pandas as pd
from settings import settings


output_file_train_aux = settings['output_file_train'] + "_aux"
output_file_train = settings['output_file_train']
output_file_test_aux = settings['output_file_test'] + "_aux"
output_file_test = settings['output_file_test']

#os.system("for filename in $(ls -lah data/images/ | grep ' 0 ' | cut -d ' ' -f12); do rm data/images/$filename; done;")

available_images = os.listdir(settings['images_path'])


val_sample = 0.3




df = pd.read_csv(settings['processed_labels_csv'], header = None)


unique_videos = df[2].unique()

valid_samples = int(unique_videos.shape[0] * 0.3)
valid_hashes = unique_videos[:valid_samples]
train_hashes = unique_videos[valid_samples:]


df_train = df[df[2].isin(train_hashes)]
df_valid = df[df[2].isin(valid_hashes)]


f = open(output_file_train_aux, 'w')
for i in df_train.index:
    hash, labels, video = df_train.ix[i]
    f.write(settings['images_path'] + "/" + hash + ".jpg," + str(' '.join(labels[1:-1].lstrip().rstrip().split(None))) + "," + str(video) + "\n")
f.close()

f = open(output_file_test_aux, 'w')
for i in df_valid.index:
    hash, labels, video = df_valid.ix[i]
    f.write(settings['images_path'] + "/" + hash + ".jpg," + str(' '.join(labels[1:-1].lstrip().rstrip().split(None))) + "," + str(video) + "\n")
f.close()

####
## This step is necessary to be sure that all the images we write in the training files do actually exist in the images folder!
####

def filter_datafiles(input_file, output_file, available_images):

    f = open(input_file)
    lines = f.readlines()
    f.close()
    '''
    accepted_output = []
    for line in lines:
        name = line[:-1].split(",")[0].split("/")[-1]
        try:
            _ = available_images.index(name)
            accepted_output.append(line)
        except:
            pass
    '''
    accepted_output = lines
    f = open(output_file, 'w')
    for line in accepted_output:
        f.write(line)
    f.close()

filter_datafiles(output_file_train_aux, output_file_train, available_images)
filter_datafiles(output_file_test_aux, output_file_test, available_images)
