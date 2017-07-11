from __future__ import unicode_literals


import os
import random
import subprocess

DATASET_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
SEPARATOR = '\t\t\t'

def get_review(file_name):
    f = open(file_name, encoding='latin1').read()
    return f


def get_reviews(directory, sentiment):
    files = os.listdir(directory)
    arr= []
    for f in files:
        arr.append('{0}{1}{2}'.format(get_review(directory + f), SEPARATOR, sentiment))
    
    return arr


def create_tsv(data_type):
    data = []
        
    data += get_reviews(directory='dataset/{0}/pos/'.format(data_type), sentiment=1)
    data += get_reviews(directory='dataset/{0}/neg/'.format(data_type), sentiment=0)
    
    # Shufle the data
    random.shuffle(data)
    
    tsv_file = open('{0}.tsv'.format(data_type), 'w', encoding='latin1')
    
    # Set the tsv file header
    tsv_file.write('Review{0}Sentiment\n'.format(SEPARATOR))
    
    for rev in data:
        tsv_file.write(rev + '\n')
        
    tsv_file.close()

# Downlod the dataset
if not os.path.isdir("./dataset"):
    os.system('wget {0}'.format(DATASET_URL))
    os.system('tar -xvzf aclImdb_v1.tar.gz')
    os.system('mv aclImdb/ dataset')


create_tsv('train')
create_tsv('test')
