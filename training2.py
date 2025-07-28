
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
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

df = pd.read_csv('coords2.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value
print("--------------------Random state 0 ------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(penalty=l1)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
print(fit_models)
'''
#fit_models['rc'].predict(X_test)
#--------------------------------------------------------------------------------------------
print("-----------------------cross validation start---------------")

''''scores = cross_val_score(fit_models['lr'], X, y, cv=5)'''


'''------------------------------ LR--------------------------'''
# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = OneVsRestClassifier(LogisticRegression())
# Convert target variable to one-hot encoded format
y_train_one_hot = label_binarize(y_train, classes=np.unique(y))
y_test_one_hot = label_binarize(y_test, classes=np.unique(y))

# Create and fit a logistic regression model
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_prob = model.predict_proba(X_test)

# Compute the false positive rate (fpr) and true positive rate (tpr) for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(len(np.unique(y))):
    plt.plot(fpr[i], tpr[i], label='ROC curve (class %d, AUC = %0.2f)' % (i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-All)')
plt.legend(loc="lower right")
plt.show()
'''
scores = cross_val_score(model, X, y, cv=5)
c = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)
print(" cross model validation : ",c)
'''
'''
scores2 = cross_val_score(l2_model, X, y, cv=5)
scores1 = cross_val_score(l1_model, X, y, cv=5)
'''
'''------------------------------------------------------------------------------'''
'''
c1 = "Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std()*2)
c2 = "Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std()*2)
print("l1 model cross validation: ",c1)
print("l2 cross model validation : ",c2)
'''
print("-----------------------cross validation end ---------------")
'''------------------------------l1,l2 regularization end --------------------------'''



'''#---------------------------------------knn-----------------------------
print("-------------------------------knn------------------------")
knn = KNeighborsClassifier(n_neighbors=5)
# fit the model to the training data
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)

print(knn)
print('Accuracy of KNN with k:5 =', f"{accuracy:.5%}")
print('training Accuracy of KNN with k:5 =', f"{train_accuracy:.5%}")

print("------------------------------knn end-------------------------")

'''
'''#------------------------------------------------models accuracy comparison-----
accuracies = []
models = ["LR","RC","RF","GB"]
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    f1 = f1_score(y_test, yhat, average='macro')
    precision = precision_score(y_test, yhat,average='macro')
    recall = recall_score(y_test, yhat,average='macro')
    accuracy = accuracy_score(y_test, yhat)

    # Print the evaluation metrics
    print("F1 score: {:.3f}".format(f1))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Accuracy: {:.3f}".format(accuracy))

    cm = confusion_matrix(y_test, yhat)
    #print(cm)
    accuracies.append(accuracy_score(y_test, yhat))
    print(algo, f"{accuracy_score(y_test, yhat):.5%}")
    print(algo, f" training accuracy : {accuracy_score(y_train, y_train_pred):.5%}")
    print("-----------------------------------------------------")
'''    
#ytrue = fit_models['lr'].predict(X_test)
#print(y_test)
#---------------------------------------dump models into pkl file to use for testing------------------
'''with open('body_languageLR1.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)
'''  
'''with open('body_languageKNN.pkl', 'wb') as f:
    pickle.dump(knn, f)
'''
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
'''
plt.bar(models, accuracies)
plt.ylim(0.95, 1.0)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.show()
'''

