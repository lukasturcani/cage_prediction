# Author: Lukas Turcani

# usage: ./make_database dirpath

# Go through all the stk population dump files in the folder
# provided as the first argument, i.e. "dirpath", and extract the
# properties of the cages held in them into SQL databse.

for db in $1/*.json
do
	python make_database.py $db
done
