#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:12:44 2017

@author: ryad
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the data set 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t',quoting = 3)

#Cleaning the Texts 
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer



text_cleaned = []


for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]' ,' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    text_cleaned.append(review)
    
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
countvectorizer = CountVectorizer()
X = countvectorizer.fit_transform(text_cleaned).toarray()
y = dataset.iloc[:,1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
# Predecting the Test set results 
y_pred = classifier.predict(X_test)
print(y_pred)

#Making the consfusion matrix
from sklearn.metrics import confusion_matrix
cm_pred = confusion_matrix(y_test,y_pred)
cm_train = confusion_matrix(y_train,classifier.predict(X_train))

accuracy = (cm_pred[0,0]+cm_pred[1,1])/(cm_pred[0,0]+cm_pred[1,1]+cm_pred[0,1]+cm_pred[1,0])
precision = (cm_pred[1,1])/(cm_pred[1,1]+cm_pred[0,1])
recall = (cm_pred[1,1])/(cm_pred[1,1]+cm_pred[1,0])
F1_score =2*precision*recall/(precision+recall)

