import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset from CSV file
df = pd.read_csv('coords2.csv')

# Separate features and labels
X = df.drop('class', axis=1) # features
y = df['class'] # target value

# Encode the labels if they are not already encoded
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scatter plot of two features (change 'feature1' and 'feature2' with actual feature names)
feature1 = 'x1'
feature2 = 'x32'
plt.scatter(X[feature1], X[feature2], c=y)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('Scatter Plot of Features')
plt.show()

# Learning curves to analyze overfitting
train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Validation Accuracy')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()