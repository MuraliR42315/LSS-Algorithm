import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
fp = pd.read_csv("400movies.csv")
data = fp.values
X= data[:,1:5]
X=np.asarray(X,dtype=np.float64)
#print (X)
Y = data[:, -1]
print (Y)
#Y=Y*100
#print (Y)
Y=np.asarray(Y,dtype=np.int64)
knnn=[]
for seed in range(10,100,10):
	c=[]
	f1=[]
	p=[]
	r=[]
	a=[]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed)
	#print (X_train,Y_train)
#print (X_test)
	clf= RandomForestClassifier(n_estimators=3, max_depth=None, min_samples_split=10, random_state=0)
	kn=kn = KNeighborsClassifier(n_neighbors=2)
	sv = SVC()
	lr = LogisticRegression()
	dtc= DecisionTreeClassifier()
	abc=AdaBoostClassifier()
	clf.fit(X_train, Y_train)
	y_pred = clf.predict(X_test)
	#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)
	#print (score)
	c.append('RF')
	a.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))
#print f1,ps, rs



	kn.fit(X_train, Y_train)
	y_pred = kn.predict(X_test)
#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)
#print (score)
	c.append('KNN')
	a.append(score)
	knnn.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))
	from sklearn.metrics import confusion_matrix

	y_pred = kn.predict(X_test)
	cm = confusion_matrix(Y_test, y_pred)

	cm=pd.DataFrame(cm, index=['Highly Readable', 'Readable','less Readable'],
                 columns=['Highly Readable', 'Readable','less Readable'])
	print (cm)
	sv.fit(X_train, Y_train)
	y_pred = sv.predict(X_test)
	
#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)
#print (score)
	c.append('SVM')
	a.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))


	lr.fit(X_train, Y_train)
	y_pred = lr.predict(X_test)
#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)
	c.append('LR')
	a.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))


	abc.fit(X_train, Y_train)
	y_pred = abc.predict(X_test)
#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)	
	c.append('ABC')
	a.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))


	dtc.fit(X_train, Y_train)
	y_pred = dtc.predict(X_test)
#print len(y_pred)
	score = metrics.accuracy_score(Y_test, y_pred)
	c.append('DTC')
	a.append(score)
	f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
	p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
	r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))
	print (c)
	print(a)
	print(f1)
	print(p)
	print(r)

	fig, ax = plt.subplots()
	index = np.arange(len(a))
	bar_width = 0.15
	opacity = 0.5
	rects1 = plt.bar(index, a, bar_width,
       	         alpha=opacity,
                 color='b',
                 label='Accuracy')
 
	rects2 = plt.bar(index + bar_width, p, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')

	rects3 = plt.bar(index +2*bar_width, r, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Recall')

	rects4 = plt.bar(index +3*bar_width, f1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='F1 Score')	


	plt.xlabel('Classifiers')
	plt.ylabel('Performance')
	plt.title('Performance of Classifiers for Grade level identification')
	plt.xticks(index + bar_width, ('RF', 'KNN', 'SVM', 'LR','ABC','DTC'))
	plt.legend()
	plt.show()

scoring = 'accuracy'
models = []	
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF',RandomForestClassifier()))
models.append(('ABC',AdaBoostClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=3, random_state=70)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
		#if (name=='KNN'):
		#	knnn.append(cv_results.mean())
fig = plt.figure()
fig.suptitle('10-fold Cross Validation Scores of Classifiers for predicting Grade level')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

scoring = 'accuracy'
models = []	
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF',RandomForestClassifier()))
models.append(('ABC',AdaBoostClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=3, random_state=30)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
		#if (name=='KNN'):
		#	knnn.append(cv_results.mean())
fig = plt.figure()
fig.suptitle('10-fold Cross Validation Scores of Classifiers for predicting Grade level')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

 # naming the x axis
plt.xlabel('Seed Value')
# naming the y axis
plt.ylabel('Accuracy')
 
	# giving a title to my graph
plt.title('Accuracy for different seed values')
 
# function to show the plot

y=range(10,100,10)
plt.plot(y,knnn,'bo')

plt.show()




'''
X= data[:,1:4]
Y = data[:, -2]
#print X
print Y
seed = 54
a=[]
f1=[]
p=[]
r=[]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed)
#print X_train,Y_train
clf= RandomForestClassifier(n_estimators=3, max_depth=None, min_samples_split=10, random_state=0)
kn=kn = KNeighborsClassifier(n_neighbors=2)
sv = SVC()
lr = LogisticRegression()
dtc= DecisionTreeClassifier()
abc=AdaBoostClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
print score
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))

kn.fit(X_train, Y_train)
y_pred = kn.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))

sv.fit(X_train, Y_train)
y_pred = sv.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))

lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))
	
abc.fit(X_train, Y_train)
y_pred = abc.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))

dtc.fit(X_train, Y_train)
y_pred = dtc.predict(X_test)
#print len(y_pred)
score = metrics.accuracy_score(Y_test, y_pred)
a.append(score)
f1.append(metrics.f1_score(Y_test, y_pred, average='weighted'))
p.append(metrics.precision_score(Y_test, y_pred,average='weighted'))
r.append(metrics.recall_score(Y_test, y_pred, average='weighted'))

print a,f1,p,r
fig, ax = plt.subplots()
index = np.arange(len(a))
bar_width = 0.15
opacity = 0.5
rects1 = plt.bar(index, a, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
 
rects2 = plt.bar(index + bar_width, p, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')

rects3 = plt.bar(index +2*bar_width, r, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Recall')

rects4 = plt.bar(index +3*bar_width, f1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='F1 Score')	


plt.xlabel('Classifiers')
plt.ylabel('Performance')
plt.title('Performance of Classifiers for Predicting Readability')
plt.xticks(index + bar_width, ('RF', 'KNN', 'SVM', 'LR','ABC','DTC'))
plt.legend()
plt.show()


scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF',RandomForestClassifier()))
models.append(('ABC',AdaBoostClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=3, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
fig = plt.figure()
fig.suptitle('10-fold Cross validation for Classifier in predicting Readability')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''
