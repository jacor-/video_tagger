##
## We split the samples prepared by the previous script in train and validation and store the resulting datasets in two different files (caffe style)
## 
##
import os
import pandas as pd


input_file_train = 'data/files/train.txt'
output_file_train = 'data/files/filtered_train.txt'
input_file_test = 'data/files/val.txt'
output_file_test = 'data/files/filtered_val.txt'

#os.system("for filename in $(ls -lah data/images/ | grep ' 0 ' | cut -d ' ' -f12); do rm data/images/$filename; done;")

available_images = os.listdir('data/images')


val_sample = 0.3




df = pd.read_csv('data/files/prepared_dataset.csv', header = None)
unique_videos = df[2].unique()

valid_samples = int(unique_videos.shape[0] * 0.3)
valid_hashes = unique_videos[:valid_samples]
train_hashes = unique_videos[valid_samples:]


df_train = df[df[2].isin(train_hashes)]
df_valid = df[df[2].isin(valid_hashes)]


f = open(input_file_train, 'w')
for i in df_train.index:
    hash, labels, video = df_train.ix[i]
    f.write("data/images/" + hash + ".jpg," + str(' '.join(labels[1:-1].lstrip().rstrip().split(None))) + "," + str(video) + "\n")
f.close()

f = open(input_file_test, 'w')
for i in df_valid.index:
    hash, labels, video = df_valid.ix[i]
    f.write("data/images/" + hash + ".jpg," + str(' '.join(labels[1:-1].lstrip().rstrip().split(None))) + "," + str(video) + "\n")
f.close()






def filter_datafiles(input_file, output_file, available_images):

    f = open(input_file)
    lines = f.readlines()
    f.close()

    accepted_output = []
    for line in lines:
        name = line[:-1].split(",")[0].split("/")[-1]
        try:
            _ = available_images.index(name)
            accepted_output.append(line)
        except:
            pass

    f = open(output_file, 'w')
    for line in accepted_output:
        f.write(line)
    f.close()




filter_datafiles(input_file_train, output_file_train, available_images)
filter_datafiles(input_file_test, output_file_test, available_images)
