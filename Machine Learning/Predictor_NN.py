print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

#pre processing
#le = preprocessing.LabelEncoder()
#Y = Y.apply(le.fit_transform)

#spliting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Training the neural network
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
mlp.fit(X_train, Y_train)

#Testing the neural network
predictions = mlp.predict(X_test)
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))