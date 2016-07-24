
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import datasets, neighbors, metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn import cross_validation

#open the dataset file and read form it the data
data=[]
f = open('/home/aearm/Desktop/IIS/Assignment2/DataSet/dataset.txt','r')
for line in f:
	currentline = line.split(",")

	if currentline[0]=='NEUTRAL':
		currentline[0]=1

	elif currentline[0]=='NEGATIVE':
		currentline[0]=0

	else:
		currentline[0]=2
	

	del currentline[-1]
	
	for i in range (len (currentline)):
		currentline[i]=float(currentline[i])	
		
	data.append(currentline)
# PCA for visualization of the data
n_samples=len(data)
ndata=np.reshape(data,(n_samples,-1))

pca=PCA(n_components=2)
X_trans=pca.fit_transform(ndata)

#tsne = TSNE(n_components=2, init='pca', random_state=0)
#X_trans = tsne.fit_transform(ndata)
plt.scatter(X_trans[:,0],X_trans[:,1],color='red',s=2*100,marker='^',alpha=0.4)

#plt.scatter(X_trans[:,0],X_trans[:,1],X_trans[:2])
plt.show()



def data_spliting_classification (fnDigits,fnData,nSamples,percentSplit):
	

	n_trainSamples = int(nSamples*percentSplit)
	
	print n_trainSamples
	print fnData
	trainData   = fnData[:n_trainSamples,:]
	#print ("train data :\n%s" % trainData)
	trainLabels =[i[0] for i in trainData]
	
	testData    = fnData[n_trainSamples:,:]
	#print ("testData:\n%s"%testData)
	expectedLabels  = [i[0] for i in testData]
	
	n_neighbors = 10	
	kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
	# trains the model
	kNNClassifier.fit(trainData, trainLabels)
	predictedLabels = kNNClassifier.predict(testData)
	
	#Display classifier results
	
	print ("predicted labels :\n%s" %predictedLabels)
	print("Classification report for classifier %s:\n%s\n"
          % ('k-NearestNeighbour', metrics.classification_report(expectedLabels, predictedLabels)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expectedLabels, predictedLabels))
	print('holdOut :: Done.')

def kFoldCrossValidation(fnDigits,fnData,kFold=5):
    
    # k-NearestNeighbour Classifier instance
    n_neighbors = 15
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    
    scores = cross_validation.cross_val_score(kNNClassifier, fnData, [i[0] for i in fnDigits], cv=kFold)
    
    print("the cross validatin scores:\n%s" %scores)
    
    print('kFoldCrossValidation :: Done.')


n_samples=len(data)

data_spliting_classification(data,ndata,n_samples,0.7)
kFoldCrossValidation(data,ndata,5)