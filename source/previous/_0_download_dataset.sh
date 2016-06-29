## This script will query the database and download all the images in a folder. 
## TODO: improve the last part to retry in case we cannot download a specific image

# In case there are some files in the destination folders, we delete them
rm -rf data/files;
rm -rf data/images;

mkdir data/files;
mkdir data/images;


# We call the Vilynx database and collect the whole dataset. Dirty, right? :)
psql postgresql://$PG_USER@$PG_HOST:5432/mldb -A -F '#' -c "SELECT frames.hash, array_agg(tags.tag), frames.vid FROM frames INNER JOIN tags on tags.vid = frames.vid WHERE category = 's' GROUP BY frames.hash, frames.vid limit 30000" -o data/files/aux.txt;
cat data/files/aux.txt | tail -n +2 | head -n -1 > data/files/raw_data.txt;
rm data/files/aux.txt;

## We iterate over all the files and download them.
## TODO: this is a really bad practice since we do not control possible errors when downloading images. We should need to do something like retry when an image cannot be downloaded.
#for IM_NAME in $(cat data/files/raw_data.txt | cut -d '#' -f 1); do
#   echo $IM_NAME;
#   wget https://public.vilynx.com/$IM_NAME/pro69.viwindow.jpg -O data/images/$IM_NAME.jpg 2> /dev/null > /dev/null;
#done;



