import pandas as pd
from sklearn import preprocessing
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from io import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data

# Read the mammographic masses dataset with proper column names and missing value handling
masses_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])

masses_data.loc[(masses_data['age'].isnull()) |
              (masses_data['shape'].isnull()) |
              (masses_data['margin'].isnull()) |
              (masses_data['density'].isnull())]

masses_data.dropna(inplace=True)
masses_data.describe()

all_features = masses_data[['age', 'shape',
                             'margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

numpy.random.seed(1234)

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)

clf= DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set
clf.fit(training_inputs, training_classes)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())  

print(clf.score(testing_inputs, testing_classes))