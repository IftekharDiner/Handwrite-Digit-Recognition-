import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").values
print(data)
data2 = pd.read_csv("test.csv").values
print(data2)

###########################


#data = pd.read_csv("datasets/train.csv").as_matrix()
data = pd.read_csv("train.csv").values
testdata = pd.read_csv("test.csv").values
clf=DecisionTreeClassifier()


###############


xtrain = data[0:42001,1:]
train_label = data[0:42001,0] 

clf.fit(xtrain,train_label)

#testing data
xtest=testdata[28000:,1:]
actual_label = testdata[28000:,0]

#first to show the match database not for final
d = xtest[0]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[0]] ))
pt.show() 

p = clf.predict(xtest)
count = 0
for i in range(0,14000):
	count += 1 if p[i]==actual_label[i] else 0
print("Accuracy=", (count/14000)*100)

#################################


import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").values

clf=DecisionTreeClassifier()

#training Datasets
xtrain = data[0:21000,1:]
train_label = data[0:21000,0] 

clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)
count = 0
for i in range(0,21000):
	count += 1 if p[i]==actual_label[i] else 0
print("Accuracy=", (count/21000)*100)

#################

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").values

clf=DecisionTreeClassifier()

#training Datasets
xtrain = data[0:21000,1:]
train_label = data[0:21000,0] 

clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)
count = 0
for i in range(0,21000):
	count += 1 if p[i]==actual_label[i] else 0
print("Accuracy=", (count/21000)*100)



