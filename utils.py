# -*- coding: utf-8 -*-

import re
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

def clean_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review


def get_corpus(data):
    corpus = []
    for i in range(0, len(data)):
        review = clean_text(data['Review'][i])
        corpus.append(review)
    return corpus
