#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import csv
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



raw_data_train = pd.read_csv('training.csv')
raw_data_train.set_index('article_number', inplace=True)
X_train_temp = raw_data_train['article_words'].to_numpy()
y_train = raw_data_train['topic'].to_numpy()
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train_temp)
ros = RandomOverSampler(random_state=42)
X_temp, y_temp = ros.fit_resample(X_train, y_train)
X_rs, X_test, y_rs, y_test = train_test_split(X_temp, y_temp, random_state=0)


rfc_model = RandomForestClassifier()
rfc_model.fit(X_rs, y_rs)
predicted_y_rfc = rfc_model.predict(X_test)
print("----===RFC===----")
#printPreview(y_test,predicted_y)
#print(rfc_model.predict_proba(X_test))
print(accuracy_score(y_test, predicted_y_rfc))
print(classification_report(y_test, predicted_y_rfc))


scores_RFC = cross_val_score(rfc_model, X_temp, y_temp, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_RFC.mean(), scores_RFC.std() * 2))





