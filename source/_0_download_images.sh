## This script will query the database and download all the images in a folder. 
## TODO: improve the last part to retry in case we cannot download a specific image

# In case there are some files in the destination folders, we delete them
#rm -rf data/files;
#rm -rf data/images;

#mkdir data/files;
#mkdir data/images;


## We iterate over all the files and download them.
#for IM_NAME in $(cat base_network/VictorData/dataset_30c.csv | cut -d '#' -f 1); do
#   echo $IM_NAME;
#   wget https://public.vilynx.com/$IM_NAME/pro69.viwindow.jpg -O data/images/$IM_NAME.jpg 2> /dev/null > /dev/null;
#done;

#Retry X times the images which has not been downloaded properly
ntries=10
for i in $(seq 1 $ntries);
    echo $ntries
	for IM_NAME in in $(ls -lah data/images/ | grep ' 0 ' | cut -d ' ' -f12); do
   		echo $IM_NAME;
        wget https://public.vilynx.com/$IM_NAME/pro69.viwindow.jpg -O data/images/$IM_NAME.jpg 2> /dev/null > /dev/null;
    done;
done;

#We delete images with 0 bytes
for filename in $(ls -lah data/images/ | grep ' 0 ' | cut -d ' ' -f12); do rm data/images/$filename; done;



