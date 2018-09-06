# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:19:20 2018

@author: Mauricio Mani
"""

"""I am using scikit-learn library to find DECISION RULES from the decision trees.
So we can make decisions"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

# This is a simulated dataset
df = pd.read_csv("C:/Users/mauri/Desktop/Big Data/kaggle/HR_kaggle.csv")
df.head()
# Size of the dataset
df.shape
df.dtypes
# There are no null values 
df.isnull().sum()
# create dummy variables
""" When modeling is very important to preprocess the data. 
However right know I will just use the model I was given with. """
df = pd.get_dummies(df, columns = ['salary', 'sales'], drop_first = True)
x = df.drop(labels = 'left', axis = 1)
y = df['left']
feature_name = x.columns
#Train and test to check out the score of our model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35)
""" When modeling, picking an easy algorithm is way better than a more colpex.
Just follow the Occam's Razor or lex parsimonaie law. In other words, the simplest solution tends 
to be the right one. 
So if we have a simple hypothesis we should select one with simple assumptions.
This of course it is not an irrefutable principle.
That is why we will select as less complexity as possible. 
 In other words, a small tree (few nodes) might work well. It will also be more simple to extract some rules."""
# Fit the decision tree algorithm
clf = DecisionTreeClassifier(max_depth = 5)
clf.fit(x_train, y_train)
# Print the score of the model
print(clf.score(x_test, y_test))
tree.export_graphviz(clf, out_file='C:/Users/mauri/Desktop/d_tree.dot', feature_names=feature_name, class_names = ["left", "in"])
""" It will also be relevant to understant how important is each feature to the model we built.
We can say that those variables are important to the decision making of the employees. """
importances = clf.feature_importances_
rel = [e for e in importances if e > 0.001]
indices = np.argsort(rel)
plt.figure()
plt.title('Importance of the features', fontdict = {'fontsize': 20, 'weight': 'bold','alpha': 0.67})
plt.barh(range(len(indices)), importances[indices], color='#5CD1BD', align='center')
plt.yticks(range(len(indices)), feature_name[indices])
plt.xlabel('Relative Importance')





