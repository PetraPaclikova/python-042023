import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
import numpy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

data = pandas.read_csv("bodyPerformance.csv")
print(data.head())

feature_cols = ["age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce"]
X = data[feature_cols]
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


categorical_columns = ["gender"]
numeric_columns = ["age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce"]


encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()
others = data[numeric_columns].to_numpy()

X = numpy.concatenate([encoded_columns, others], axis =1)

clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()
print(accuracy_score(y_test,y_pred))


# dot_data = StringIO()
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["A", "B", "C", "D"])
# graph.write_png('tree.png')

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=feature_cols, class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')

print("Accuracy skore u modelu DecisionTreeClassifier je 43,78%")

# KNearest neighbour model (data jsem nechala stejne transformovne jako u predchziho modelu.)


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()
print(accuracy_score(y_test, y_pred))
print("Accuracy skore u modelu KNeighnoursClassifier je 39,6%")