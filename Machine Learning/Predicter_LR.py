print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#reading data from the file
fh = open("data_training.txt","r");
data = fh.readlines()

#initializing the array
X = np.zeros(( len(data) , len(data[0].split())-1 ))
Y = np.zeros(len(data))

#populating the array with training data
numberOfData = len(data)
numberOfFeatures = len(data[0].split()) - 1
for i in range(numberOfData):
    for j in range(numberOfFeatures + 1):
        if j != numberOfFeatures:
            X[i][j] = float( data[i].split()[j] )
        else:
            Y[i] = float( data[i].split()[j] ) - 1.0

#spliting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

logreg = linear_model.LogisticRegression(C=1e5)
# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)
print(logreg.score(X_test, Y_test))