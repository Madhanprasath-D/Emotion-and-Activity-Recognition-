import numpy as np
import csv
import os
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv('data.csv')
# print(df.shape)
x = df.drop('class',axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)

pipeline = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

fit_model = {}
for al, pipeline in  pipeline.items():
    model = pipeline.fit(x_train,y_train)
    fit_model[al] = model


for i, j in fit_model.items():
    yhat = j.predict(x_test)
    #print(i,accuracy_score(y_test,yhat))

with open('model.pkl', mode='wb') as f:
    pickle.dump(fit_model['rf'],f)
