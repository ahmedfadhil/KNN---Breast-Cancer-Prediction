import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# Model Features
X = np.array(df.drop(['class'], 1))
# Model labels
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

cls = neighbors.KNeighborsClassifier()

cls.fit(X_train, y_train)

accuracy = cls.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = cls.predict(example_measures)

# print(confusion_matrix(y_test, example_measures))
# print(classification_report(y_test, example_measures))
print(prediction)
