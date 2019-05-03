#%%
import numpy as np
import random
from numpy.linalg.linalg import pinv
import pandas as pd
from sklearn.model_selection import train_test_split



class LoadAndScaleData:
    def __init__(self,filePath):
        self.data = np.loadtxt(filePath,delimiter=',',usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
        self.labels = np.loadtxt(filePath,delimiter=',',dtype='S',usecols=(1))
        x, = self.labels.shape
        self.rLabels = np.zeros(shape = (0,2))
        for i in range(0,x):
            if self.labels[i].decode('UTF-8') == 'M':
                self.rLabels = np.vstack([self.rLabels,[1,0]])
            if self.labels[i].decode('UTF-8') == 'B':
                self.rLabels = np.vstack([self.rLabels,[0,1]])

    def scale(self):
        mat = np.asmatrix(self.data)
        height,width = mat.shape
        for i in range(0,width):
            minimum = np.min(mat[:,i])
            maximum = np.max(mat[:,i])
            for k in range(0,height):
                mat[k,i] = (mat[k,i] - minimum)/(maximum - minimum)
        return mat, self.rLabels

class RBFNetwork():
    def __init__(self, pTypes,scaledData,labels):
        self.pTypes = pTypes
        self.protos = np.zeros(shape=(0,31))
        self.scaledData = scaledData
        self.spread = 0
        self.labels = labels
        self.weights = 0
        self.indexM = 0
        self.indexB = 0
        self.input_train = 0
        self.input_test = 0
        self.output_train = 0
        self.output_test = 0

    def splitData(self):
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(self.scaledData,
                                                                                                  self.labels,
                                                                                                  stratify=self.labels,
                                                                                                  test_size=0.2)
        self.input_train = self.input_train.tolist()
        self.input_test = self.input_test.tolist()
        self.output_train = self.output_train.tolist()
        #self.output_test = self.output_test.tolist()
        return self.input_test


    def generatePrototypes(self):
        self.indexM = [i for i, x in enumerate(self.output_train) if x == [1, 0]]
        self.indexB = [i for i, x in enumerate(self.output_train) if x == [0, 1]]
        groupM = np.random.choice(self.indexM, size=self.pTypes)
        groupB = np.random.choice(self.indexB,size=self.pTypes)
        #print("groupM", groupM)
        #print("groupB", groupB)
        self.protos = np.vstack([self.protos,self.scaledData[groupM,:],self.scaledData[groupB,:]])
        return self.protos

    def sigma(self):
        dTemp = 0
        for i in range(0,self.pTypes*2):
            for k in range(0,self.pTypes*2):
                dist = np.square(np.linalg.norm(self.protos[i] - self.protos[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp/np.sqrt(self.pTypes*2)

    def train(self):
        self.splitData()
        self.generatePrototypes()
        self.sigma()
        hiddenOut = np.zeros(shape=(0,self.pTypes*2))
        for item in self.scaledData:
            out=[]
            for proto in self.protos:
                distance = np.square(np.linalg.norm(item - proto))
                neuronOut = np.exp(-(distance)/(np.square(self.spread)))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut,np.array(out)])
        #print(hiddenOut)
        self.weights = np.dot(pinv(hiddenOut),self.labels)
        #print(self.weights)

    def test(self):
        class_actual = []
        class_predicted = []
        for i in range(len(self.input_test)):
            data = self.input_test[i]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data-proto))
                neuronOut = np.exp(-(distance)/np.square(self.spread))
                out.append(neuronOut)
            netOut = np.dot(np.array(out),self.weights)
            #print('---------------------------------')
            #print(netOut)
            output_class = 0
            if (netOut.argmax(axis=0) + 1 == 1):
                output_class = 'Malignant'
            elif (netOut.argmax(axis=0) + 1 == 2):
                output_class = 'Benign'
            #print('Class is ', output_class)
            class_actual.append(netOut.argmax(axis=0) + 1)
            if (self.output_test[i].argmax(axis=0) +1 == 1):
                output_class = 'Malignant'
            elif (self.output_test[i].argmax(axis=0) +1 == 2):
                output_class = 'Benign'
            #print('Given Class ',output_class)
            class_predicted.append(self.output_test[i].argmax(axis=0) +1)
        y_actu = pd.Series(class_actual, name='Actual')
        y_pred = pd.Series(class_predicted, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        #return str(df_confusion), str(netOut), str("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0])/len(self.output_test) * 100))
        return str("Data"), str("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0])/len(self.output_test) * 100)), str(output_class)
        #print(df_confusion)
        #print("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0])/len(self.output_test) * 100))


    def testTest(self, dataTest, dataTestLabels):
        #print("Testing data")
        class_actual = []
        class_predicted = []
        for i in range(len(dataTest)):
            data = dataTest[i]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data - proto))
                neuronOut = np.exp(-(distance) / np.square(self.spread))
                out.append(neuronOut)
            netOut = np.dot(np.array(out), self.weights)
            #print('---------------------------------')
           # print(netOut)
            output_class = 0
            if (netOut.argmax(axis=0) + 1 == 1):
                output_class = 'Malignant'
            elif (netOut.argmax(axis=0) + 1 == 2):
                output_class = 'Benign'
          #  print('Class is ', output_class)
            class_actual.append(netOut.argmax(axis=0) + 1)
            if (dataTestLabels[i].argmax(axis=0) + 1 == 1):
                output_class = 'Malignant'
            elif (dataTestLabels[i].argmax(axis=0) + 1 == 2):
                output_class = 'Benign'
         #   print('Given Class ', output_class)
            class_predicted.append(dataTestLabels[i].argmax(axis=0) + 1)
        y_actu = pd.Series(class_actual, name='Actual')
        y_pred = pd.Series(class_predicted, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        #return str(df_confusion), str(netOut), str("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0]) / len(dataTest) * 100))
        return ("Testing data"), str("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0]) / len(dataTest) * 100)), str(output_class)
        #print(df_confusion)
        #print("The accuracy is {0:.2f} %".format(len(np.where(np.equal(y_actu, y_pred))[0]) / len(dataTest) * 100))

    def cc(self):
        self.c = 11

    def pron(self):
        return self.c

    def printing(self):
        return self.df_confusion


def hh():
    x = 2
    return x


data = LoadAndScaleData('data_train.csv')
scaledData, label = data.scale()
#print(scaledData[0])
network = RBFNetwork(5,scaledData,label)
network.train()
a = network.test()

dataT = LoadAndScaleData('data_test.csv')
dataTest, dataTestLabels = dataT.scale()
b = network.testTest(dataTest, dataTestLabels)





