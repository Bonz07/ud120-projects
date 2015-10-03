#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'fraction_emails_from_POI', 'fraction_emails_to_POI',
                 'shared_receipt_with_poi', 'restricted_stock', 'exercised_stock_options',
                 'total_stock_value', 'expenses', 'to_messages', 'deferral_payments',
                 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'from_messages', 'other', 'long_term_incentive',
                 'director_fees']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Identify and remove outliers

###Visually inspect the names of each entry to make sure they are all employees of Enron Corportation
#for i in data_dict:
#    print i

###describe main features of data

#poi_total = 0
#print len(data_dict)
#for i in data_dict:
 #   if data_dict[i]['poi'] == True:
 #       poi_total += 1
#print poi_total

###Identify any employees that have no values for any of the features
#for i in data_dict:
#    cache = 0
#    for j in data_dict[i]:
#        if data_dict[i][j] == 'NaN':
#            cache += 1
#    if cache == 20:
#        print i

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Visualising
#features = ["salary", "bonus", "poi"]
#data = featureFormat(data_dict, features)

#for point in data:
#    salary = point[0]
#    bonus = point[1]
#    if point[2] == True:
#        poi = "b"
#    else:
#        poi = "r"
#    plot = matplotlib.pyplot.scatter( salary, bonus, color=poi )

#matplotlib.pyplot.xlabel("Salary ($10 millions)")
#matplotlib.pyplot.ylabel("Bonus ($ 10 millions)")
#matplotlib.pyplot.xlim(0,)
#matplotlib.pyplot.ylim(0,)
#matplotlib.pyplot.legend([plot], ['POI'])
#matplotlib.pyplot.xlim(0, 1200000)
#matplotlib.pyplot.ylim(0, 10000000)
#matplotlib.pyplot.show()
 

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict
for i in my_dataset:
    if my_dataset[i]['from_poi_to_this_person'] == 0 or my_dataset[i]['from_poi_to_this_person'] == 'NaN':
        fraction = 0
    else:
        fraction = float(my_dataset[i]['from_poi_to_this_person'])/float(my_dataset[i]['to_messages'])
    my_dataset[i]['fraction_emails_from_POI'] = fraction

    if my_dataset[i]['from_this_person_to_poi'] == 0 or my_dataset[i]['from_this_person_to_poi'] == 'NaN':
        fraction = 0
    else:
        fraction = float(my_dataset[i]['from_this_person_to_poi'])/float(my_dataset[i]['from_messages'])
    my_dataset[i]['fraction_emails_to_POI'] = fraction

###SelectKBest

from sklearn.feature_selection import SelectKBest

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
k=10
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:k])

final_features = ['poi']
for feature in k_best_features:
    final_features.append(feature)
    print feature, k_best_features[feature]

feature_list = final_features
print feature_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers

##Naive Bayes (no tuning)
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

##Support Vector Machines
#from sklearn.svm import SVC
#clf = SVC()

#from sklearn.grid_search import GridSearchCV
#parameters = {'kernel':('rbf', 'linear', 'poly'),'C':[1,1000], 'gamma':[0.0,1000]}
#svr = SVC()
#clf = GridSearchCV(svr, parameters).fit(features, labels)
#print clf.best_estimator_
#print clf.best_params_

##K Means Clustering
#from sklearn.cluster import KMeans
#clf = KMeans(n_clusters=2, tol=0.001)

##Decision Trees
from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.grid_search import GridSearchCV
parameters = {'criterion':('gini', 'entropy'), 'splitter':('best','random')}
svr = clf
clf = GridSearchCV(svr, parameters).fit(features, labels)
print clf.best_estimator_
print clf.best_params_


##Validation
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = []
precision = []
recall = []

for attempt in range(1000):
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3)
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    accuracy.append(accuracy_score(labels_test, predictions))
    precision.append(precision_score(labels_test, predictions))
    recall.append(recall_score(labels_test, predictions))

print "accuarcy: {}".format(mean(accuracy))
print "precision: {}".format(mean(precision))
print "recall: {}".format(mean(recall))

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)