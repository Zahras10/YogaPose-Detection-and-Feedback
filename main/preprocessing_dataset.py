import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv('coords2.csv')

X = data.drop('class', axis=1) # features
y = data['class'] # target value
# Print unique values and their counts for each column

unique_values = data['class'].unique()
unique_counts = data['class'].value_counts()
print(f"Unique Values: {unique_values}")
print(f"Value Counts:\n{unique_counts}\n")


plt.figure(figsize=(8, 6))
plt.bar(unique_values, unique_counts)

    # Set labels and title
plt.xlabel('class')
plt.ylabel("Count")
plt.title(f"Unique Values for yoga pose")

    # Rotate x-axis labels if needed
plt.xticks(rotation=90)

    # Show the plot
plt.show()


duplicate_rows = data[data.duplicated()]

if duplicate_rows.empty:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicate_rows)
'''



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling using RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

# Undersampling using RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

# Print the class distribution before and after sampling
print("Class Distribution (Before Sampling):")
print(y_train.value_counts())

print("\nClass Distribution (After Oversampling):")
print(y_train_over.value_counts())

print("\nClass Distribution (After Undersampling):")
print(y_train_under.value_counts())


unique_values1 = y_train_over.unique()
unique_counts1 = y_train_over.value_counts()
print(f"Unique Values: {unique_values1}")
print(f"Value Counts:\n{unique_counts1}\n")


plt.figure(figsize=(8, 6))
plt.bar(unique_values1, unique_counts1)

    # Set labels and title
plt.xlabel('class')
plt.ylabel("Count")
plt.title(f"Unique Values for yoga pose after over sampling")

    # Rotate x-axis labels if needed
plt.xticks(rotation=90)

    # Show the plot
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_train_over, y_train_over, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

# Fit the model using the oversampled data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
scores = cross_val_score(model, X, y, cv=5)
c = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2)
print(" cross model validation : ",c)

'''