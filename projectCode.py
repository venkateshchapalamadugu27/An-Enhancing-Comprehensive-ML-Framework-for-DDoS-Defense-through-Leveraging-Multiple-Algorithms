import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import arff as arf

file = open("final-dataset.arff")
decoder = arf.ArffDecoder()
data = decoder.decode(file, encode_nominal=True)
vals = [val[0:-1] for val in data['data']]
labels = [lab[-1] for lab in data['data']]

brac = 100000
vals = vals[:brac]
labels = labels[:brac]
print(vals)
print(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
vals, labels, test_size=0.2, random_state=0)
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual classifiers
svm_classifier = SVC(kernel='sigmoid', gamma='auto', probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
naive_bayes_classifier = GaussianNB()

voting_classifier.fit(X_train_scaled, y_train)

y_pred_voting = voting_classifier.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_pred_voting, y_test)
print("Amulgamated model accuracy:", ensemble_accuracy * 100, "%")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming y_true and y_pred are your true labels and predicted labels, respectively
cm = confusion_matrix( y_test, y_pred_voting)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.model_selection import cross_val_score
# Perform 5-fold cross-validation
cross_val_scores = cross_val_score(voting_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
# Print the cross-validation scores
print("Cross-validation scores:", cross_val_scores)
print("Mean accuracy:", np.mean(cross_val_scores))


from sklearn.metrics import accuracy_score, precision_score, f1_score
# Make predictions on the test set
y_pred_voting = voting_classifier.predict(X_test_scaled)
# Evaluate the accuracy, precision, and F1-score of the ensemble classifier
ensemble_accuracy = accuracy_score(y_pred_voting, y_test)
ensemble_precision = precision_score(y_pred_voting, y_test, average='weighted')
ensemble_f1_score = f1_score(y_pred_voting, y_test, average='weighted')
print("Amalgamated model accuracy:", ensemble_accuracy * 100, "%")
print("Amalgamated model precision:", ensemble_precision * 100, "%")
print("Amalgamated model F1-score:", ensemble_f1_score * 100, "%")