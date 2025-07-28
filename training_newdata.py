
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


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

df = pd.read_csv('coords2.csv')
df1 = pd.read_csv('coords1test.csv')
#--------df2 is testing dataset----------------
print(df.shape)
print(df1.shape)
print("---------------------------------------------------------------------------")
X = df.drop('class', axis=1) # features
y = df['class'] # target value

X1 = df1.drop('class', axis=1) # features
y1 = df1['class'] # target value

print("--------------------Random state 0 ------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', C=1.0)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
print(fit_models)

#fit_models['rc'].predict(X_test)
#--------------------------------------------------------------------------------------------





#--------------------models accuracy comparison with X1,y1 new testing dataset-----
accuracies = []
models = ["LR","RC","RF","GB"]
for algo, model in fit_models.items():
    yhat = model.predict(X1)
    f1 = f1_score(y1, yhat, average='macro')
    precision = precision_score(y1, yhat,average='macro')
    recall = recall_score(y1, yhat,average='macro')
    accuracy = accuracy_score(y1, yhat)

    # Print the evaluation metrics
    print("F1 score: {:.3f}".format(f1))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Accuracy: {:.3f}".format(accuracy))

    cm = confusion_matrix(y1, yhat)
    #print(cm)
    accuracies.append(accuracy_score(y1, yhat))
    print(algo, f"{accuracy_score(y1, yhat):.5%}")
    print("-----------------------------------------------------")
   
#ytrue = fit_models['lr'].predict(X_test)
#print(y_test)
#---------------------------------------dump models into pkl file to use for testing------------------
'''with open('body_languageLR1.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)
  


with open('body_languageRC.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'], f)
with open('body_languageRF.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
'''
with open('body_languageGB.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)
  
''' drawing graph for comparsion
'''

plt.bar(models, accuracies)
plt.ylim(0.95, 1.0)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.show()
