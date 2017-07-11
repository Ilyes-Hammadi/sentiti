# -*- coding: utf-8 -*-
import pickle

from keras.models import load_model

from utils import clean_text

# Load the CountVectorizer object
cv_file = open('count_vectorizer.pk', 'rb')
cv = pickle.load(cv_file)


# Load the trained model
classifier = load_model('classifier.h5')

def predict(text):
    """
        Predict sentiment
    """
    cleaned_text = clean_text(text)
    x = cv.transform([cleaned_text]).toarray()
    pred = classifier.predict(x)
    
    pos = float(pred[0][1])
    neg = float(pred[0][0])
    
    senti = {'POS': pos, 'NEG': neg}
    senti['SENTIMENT'] = max(senti, key=senti.get)
    return senti