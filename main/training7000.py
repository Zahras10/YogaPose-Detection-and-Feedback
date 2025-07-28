
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix # Accuracy metrics 
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

df = pd.read_csv('coords2.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

# Label encode the string class labels
print(y)

print("--------------------Random state 0 ------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', C=1.0)),
    
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
print(fit_models)

#fit_models['rc'].predict(X_test)
#--------------------------------------------------------------------------------------------
print("-----------------------cross validation start---------------")
'''
scores = cross_val_score(model, X, y, cv=5)
c = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)
print(" cross model validation : ",c)
'''

print("-----------------------cross validation end ---------------")




#------------------------------------------------models accuracy comparison-----
accuracies = []
models = ["LR","RC","RF","GB"]
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    f1 = f1_score(y_test, yhat, average='macro')
    precision = precision_score(y_test, yhat,average='macro')
    recall = recall_score(y_test, yhat,average='macro')
    accuracy = accuracy_score(y_test, yhat)

    # Print the evaluation metrics
    print("F1 score: {:.3f}".format(f1))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Accuracy: {:.3f}".format(accuracy))

    scores = cross_val_score(model, X, y, cv=5)
    c = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)
    print(f" {model} cross validation : ",c)

    cm = confusion_matrix(y_test, yhat)
    #print(cm)
    accuracies.append(accuracy_score(y_test, yhat))
    print(algo, f"{accuracy_score(y_test, yhat):.5%}")
    print("-----------------------------------------------------")
   
#ytrue = fit_models['lr'].predict(X_test)
#print(y_test)
#---------------------------------------dump models into pkl file to use for testing------------------
with open('body_languageLR7000.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)
  
'''

with open('body_languageRC.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'], f)
with open('body_languageRF.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
with open('body_languageGB.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)
  ''' 
''' drawing graph for comparsion
'''


