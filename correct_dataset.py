import os
from PIL import Image
import pandas as pd


base_dataset_filename = '/home/ubuntu/datasets_vilynx/labels/multi_frame_new_tags.csv'
out_dataset_filename = '/home/ubuntu/datasets_vilynx/labels/multi_frame_new_tags_filtered_jose.csv'
images_path = '/home/ubuntu/uploaded_images/images/'

errors = []
correct = []
for im in os.listdir(images_path):
    try:
        Image.open(images_path + im); correct.append(im)
    except:
        errors.append(im)
print([len(errors), len(correct)])

data = pd.DataFrame(map(lambda x: x.split("_")[0], correct))
data[1] = 1

data2 = data.groupby(0).sum()
data2 = data2.reset_index()

valid_videos_basehash = data2[data2[1] == 60][0].values
df_valid = pd.DataFrame(valid_videos_basehash)
dfn = pd.read_csv(base_dataset_filename, sep = ';', header = None)
dfn['videohash'] = dfn.apply(lambda x: x[0].split("_")[0], axis=1)



df_final = pd.merge(df_valid, dfn, left_on = [0], right_on = ['videohash'])
df_final = df_final[['0_y',1,2,3,4]]
df_final.to_csv(out_dataset_filename, header = False, index = False, sep = ";")


print("Out of the available images:")
print(" - %d were incorrectly downloaded" % len(errors))
print(" - %d were properly downloaded" % len(correct))
print("The original dataset contained %d videos" % dfn.shape[0])
print("The new dataset contains %d videos" % df_final.shape[0])
