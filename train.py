# -*- coding: utf-8 -*-

import pickle

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


from utils import get_corpus

# Get the data
train = pd.read_csv('train.tsv', sep='\t\t\t')
test = pd.read_csv('test.tsv', sep='\t\t\t')

# Count Vectorizer as a global variable 
cv = CountVectorizer()

# Cleaning the texts
def bag_of_words(data_train, data_test):
    corpus_train = get_corpus(data_train)
    
    # Creating the Bag of Words model
    x_train = cv.fit_transform(corpus_train).toarray()
    y_train = data_train.iloc[:, 1].values
    
    corpus_test = get_corpus(data_test)
    x_test = cv.transform(corpus_test).toarray()
    y_test = data_test.iloc[:, 1].values
    
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = bag_of_words(train[:4000], test[:1000])


# Convert integers to dummy variables
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#########################
# Building the model
########################
# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
classifier.add(Dropout(0.2))

classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.2))


# Adding the output layer
classifier.add(Dense(output_dim=2, init = 'uniform', activation = 'softmax'))


# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=32, nb_epoch=10)

classifier.save('classifier.h5')


# Save the Count Vectorizer into a pickle file
cv_file = open('count_vectorizer.pk', 'wb')
pickle.dump(cv, cv_file)
