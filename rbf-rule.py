import numpy as np
import random
from numpy.linalg.linalg import pinv
import pandas as pd
from sklearn.model_selection import train_test_split


class LoadAndScaleData:
    def __init__(self, filePath):
        self.data = np.loadtxt(filePath, delimiter=',', usecols=(
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
        self.labels = np.loadtxt(filePath, delimiter=',', dtype='S', usecols=(1))
        x, = self.labels.shape
        self.rLabels = np.zeros(shape=(0, 2))
        for i in range(0, x):
            if self.labels[i].decode('UTF-8') == 'M':
                self.rLabels = np.vstack([self.rLabels, [1, 0]])
            if self.labels[i].decode('UTF-8') == 'B':
                self.rLabels = np.vstack([self.rLabels, [0, 1]])

    def scale(self):
        mat = np.asmatrix(self.data)
        height, width = mat.shape
        for i in range(0, width):
            minimum = np.min(mat[:, i])
            maximum = np.max(mat[:, i])
            for k in range(0, height):
                mat[k, i] = (mat[k, i] - minimum) / (maximum - minimum)
        return mat, self.rLabels


class RBFNetwork:
    def __init__(self, pTypes, scaledData, labels):
        self.pTypes = pTypes
        self.protos = np.zeros(shape=(0, 30))
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
        # self.output_test = self.output_test.tolist()

    def generatePrototypes(self):
        self.indexM = [i for i, x in enumerate(self.output_train) if x == [1, 0]]
        self.indexB = [i for i, x in enumerate(self.output_train) if x == [0, 1]]
        groupM = np.random.choice(self.indexM, size=self.pTypes)
        groupB = np.random.choice(self.indexB, size=self.pTypes)
        print("groupM", groupM)
        print("groupB", groupB)
        self.protos = np.vstack([self.protos, self.scaledData[groupM, :], self.scaledData[groupB, :]])
        return self.protos

    def sigma(self):
        dTemp = 0
        for i in range(0, self.pTypes * 2):
            for k in range(0, self.pTypes * 2):
                dist = np.square(np.linalg.norm(self.protos[i] - self.protos[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp / np.sqrt(self.pTypes * 2)

    def train(self):
        self.splitData()
        self.generatePrototypes()
        self.sigma()
        hiddenOut = np.zeros(shape=(0, self.pTypes * 2))
        for item in self.scaledData:
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(item - proto))
                neuronOut = np.exp(-(distance) / (np.square(self.spread)))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut, np.array(out)])
        # print(hiddenOut)
        self.weights = np.dot(pinv(hiddenOut), self.labels)
        # print(self.weights)

    def test(self):
        class_actual = []
        class_predicted = []
        for i in range(len(self.input_test)):
            data = self.input_test[i]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data - proto))
                neuronOut = np.exp(-(distance) / np.square(self.spread))
                out.append(neuronOut)
            netOut = np.dot(np.array(out), self.weights)
            print('---------------------------------')
            print(netOut)
            output_class = 0
            if (netOut.argmax(axis=0) + 1 == 1):
                output_class = 'M'
            elif (netOut.argmax(axis=0) + 1 == 2):
                output_class = 'B'
            print('Class is ', output_class)
            class_actual.append(netOut.argmax(axis=0) + 1)
            if (self.output_test[i].argmax(axis=0) + 1 == 1):
                output_class = 'M'
            elif (self.output_test[i].argmax(axis=0) + 1 == 2):
                output_class = 'B'
            print('Given Class ', output_class)
            class_predicted.append(self.output_test[i].argmax(axis=0) + 1)
        y_actu = pd.Series(class_actual, name='Actual')
        y_pred = pd.Series(class_predicted, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        print(df_confusion)
        print("The accuracy is {0:.2f} %".format(
            len(np.where(np.equal(y_actu, y_pred))[0]) / len(self.output_test) * 100))

data = LoadAndScaleData('data.csv')
scaledData, label = data.scale()
network = RBFNetwork(5,scaledData,label)
network.train()
network.test()

#centers
ci = np.asarray(network.protos)

minm = np.zeros(shape = (30,1))
maxm = np.zeros(shape = (30,1))
minb = np.zeros(shape = (30,1))
maxb = np.zeros(shape = (30,1))
for i in range(30):
    temp1 = []
    temp2 = []
    for p in range(5):
        temp1.append(ci[p][i])
    for p in range(5,10):
        temp2.append(ci[p][i])
    minm[i] = round(min(temp1), 4)
    maxm[i] = round(max(temp1), 4)
    minb[i] = round(min(temp2), 4)
    maxb[i] = round(max(temp2), 4)

# fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

at0 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at0')
at0['m'] = fuzz.trapmf(at0.universe, [0, minm[0], maxm[0], 1])
at0['b'] = fuzz.trapmf(at0.universe, [0, minb[0], maxb[0], 1])

at1 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at1')
at1['m'] = fuzz.trapmf(at1.universe, [0, minm[1], maxm[1], 1])
at1['b'] = fuzz.trapmf(at1.universe, [0, minb[1], maxb[1], 1])

at2 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at2')
at2['m'] = fuzz.trapmf(at2.universe, [0, minm[2], maxm[2], 1])
at2['b'] = fuzz.trapmf(at2.universe, [0, minb[2], maxb[2], 1])

at3 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at3')
at3['m'] = fuzz.trapmf(at3.universe, [0, minm[3], maxm[3], 1])
at3['b'] = fuzz.trapmf(at3.universe, [0, minb[3], maxb[3], 1])

at4 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at4')
at4['m'] = fuzz.trapmf(at4.universe, [0, minm[4], maxm[4], 1])
at4['b'] = fuzz.trapmf(at4.universe, [0, minb[4], maxb[4], 1])

at5 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at5')
at5['m'] = fuzz.trapmf(at5.universe, [0, minm[5], maxm[5], 1])
at5['b'] = fuzz.trapmf(at5.universe, [0, minb[5], maxb[5], 1])

at7 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at7')
at7['m'] = fuzz.trapmf(at7.universe, [0, minm[7], maxm[7], 1])
at7['b'] = fuzz.trapmf(at7.universe, [0, minb[7], maxb[7], 1])

at6 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at6')
at6['m'] = fuzz.trapmf(at6.universe, [0, minm[6], maxm[6], 1])
at6['b'] = fuzz.trapmf(at6.universe, [0, minb[6], maxb[6], 1])

at9 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at9')
at9['m'] = fuzz.trapmf(at9.universe, [0, minm[9], maxm[9], 1])
at9['b'] = fuzz.trapmf(at9.universe, [0, minb[9], maxb[9], 1])

at8 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at8')
at8['m'] = fuzz.trapmf(at8.universe, [0, minm[8], maxm[8], 1])
at8['b'] = fuzz.trapmf(at8.universe, [0, minb[8], maxb[8], 1])

at10 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at10')
at10['m'] = fuzz.trapmf(at10.universe, [0, minm[10], maxm[10], 1])
at10['b'] = fuzz.trapmf(at10.universe, [0, minb[10], maxb[10], 1])

at11 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at11')
at11['m'] = fuzz.trapmf(at11.universe, [0, minm[11], maxm[11], 1])
at11['b'] = fuzz.trapmf(at11.universe, [0, minb[11], maxb[11], 1])

at12 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at12')
at12['m'] = fuzz.trapmf(at12.universe, [0, minm[12], maxm[12], 1])
at12['b'] = fuzz.trapmf(at12.universe, [0, minb[12], maxb[12], 1])

at13 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at13')
at13['m'] = fuzz.trapmf(at13.universe, [0, minm[13], maxm[13], 1])
at13['b'] = fuzz.trapmf(at13.universe, [0, minb[13], maxb[13], 1])

at14 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at14')
at14['m'] = fuzz.trapmf(at14.universe, [0, minm[14], maxm[14], 1])
at14['b'] = fuzz.trapmf(at14.universe, [0, minb[14], maxb[14], 1])

at15 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at15')
at15['m'] = fuzz.trapmf(at15.universe, [0, minm[15], maxm[15], 1])
at15['b'] = fuzz.trapmf(at15.universe, [0, minb[15], maxb[15], 1])

at17 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at17')
at17['m'] = fuzz.trapmf(at17.universe, [0, minm[17], maxm[17], 1])
at17['b'] = fuzz.trapmf(at17.universe, [0, minb[17], maxb[17], 1])

at16 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at16')
at16['m'] = fuzz.trapmf(at16.universe, [0, minm[16], maxm[16], 1])
at16['b'] = fuzz.trapmf(at16.universe, [0, minb[16], maxb[16], 1])

at19 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at19')
at19['m'] = fuzz.trapmf(at19.universe, [0, minm[19], maxm[19], 1])
at19['b'] = fuzz.trapmf(at19.universe, [0, minb[19], maxb[19], 1])

at18 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at18')
at18['m'] = fuzz.trapmf(at18.universe, [0, minm[18], maxm[18], 1])
at18['b'] = fuzz.trapmf(at18.universe, [0, minb[18], maxb[18], 1])

at20 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at20')
at20['m'] = fuzz.trapmf(at20.universe, [0, minm[20], maxm[20], 1])
at20['b'] = fuzz.trapmf(at20.universe, [0, minb[20], maxb[20], 1])

at21 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at21')
at21['m'] = fuzz.trapmf(at21.universe, [0, minm[21], maxm[21], 1])
at21['b'] = fuzz.trapmf(at21.universe, [0, minb[21], maxb[21], 1])

at22 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at22')
at22['m'] = fuzz.trapmf(at22.universe, [0, minm[22], maxm[22], 1])
at22['b'] = fuzz.trapmf(at22.universe, [0, minb[22], maxb[22], 1])

at23 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at23')
at23['m'] = fuzz.trapmf(at23.universe, [0, minm[23], maxm[23], 1])
at23['b'] = fuzz.trapmf(at23.universe, [0, minb[23], maxb[23], 1])

at24 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at24')
at24['m'] = fuzz.trapmf(at24.universe, [0, minm[24], maxm[24], 1])
at24['b'] = fuzz.trapmf(at24.universe, [0, minb[24], maxb[24], 1])

at25 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at25')
at25['m'] = fuzz.trapmf(at25.universe, [0, minm[25], maxm[25], 1])
at25['b'] = fuzz.trapmf(at25.universe, [0, minb[25], maxb[25], 1])

at27 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at27')
at27['m'] = fuzz.trapmf(at27.universe, [0, minm[27], maxm[27], 1])
at27['b'] = fuzz.trapmf(at27.universe, [0, minb[27], maxb[27], 1])

at26 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at26')
at26['m'] = fuzz.trapmf(at26.universe, [0, minm[26], maxm[26], 1])
at26['b'] = fuzz.trapmf(at26.universe, [0, minb[26], maxb[26], 1])

at29 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at29')
at29['m'] = fuzz.trapmf(at29.universe, [0, minm[29], maxm[29], 1])
at29['b'] = fuzz.trapmf(at29.universe, [0, minb[29], maxb[29], 1])

at28 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at28')
at28['m'] = fuzz.trapmf(at28.universe, [0, minm[28], maxm[28], 1])
at28['b'] = fuzz.trapmf(at28.universe, [0, minb[28], maxb[28], 1])

output = ctrl.Consequent(np.arange(0, 1, 0.01), 'output')
output['m'] = fuzz.trimf(output.universe, [0, 0, 0.6])
# output['idk'] = fuzz.trimf(output.universe, [0, 0.5, 1])
output['b'] = fuzz.trimf(output.universe, [0.4, 1, 1])

rule1 = ctrl.Rule(at0['m'] | at1['m'] | at2['m'] | at3['m'] | at4['m'] | at5['m'] | at6['m'] | at7['m'] | at8['m'] | at9['m'] | at10['m'] | at11['m'] | at12['m'] | at13['m'] | at14['m'] | at15['m'] | at16['m'] | at17['m'] | at18['m'] | at19['m'] | at20['m'] | at21['m'] | at22['m'] | at23['m'] | at24['m'] | at25['m'] | at26['m'] | at27['m'] | at28['m'] | at29['m'], output['m'])
rule2 = ctrl.Rule(at0['b'] | at1['b'] | at2['b'] | at3['b'] | at4['b'] | at5['b'] | at6['b'] | at7['b'] | at8['b'] | at9['b'] | at10['b'] | at11['b'] | at12['b'] | at13['b'] | at14['b'] | at15['b'] | at16['b'] | at17['b'] | at18['b'] | at19['b'] | at20['b'] | at21['b'] | at22['b'] | at23['b'] | at24['b'] | at25['b'] | at26['b'] | at27['b'] | at28['b'] | at29['b'], output['b'])
#rule3 = ctrl.Rule(at0['2'] | at1['2'] | at2['2'] | at3['2'] | at4['2'] | at5['2'] | at6['2'] | at7['2'] | at8['2'] | at9['2'] | at10['2'] | at11['2'] | at12['2'] | at13['2'] | at14['2'] | at15['2'] | at16['2'] | at17['2'] | at18['2'] | at19['2'] | at20['2'] | at21['2'] | at22['2'] | at23['2'] | at24['2'] | at25['2'] | at26['2'] | at27['3'] | at28['2'] | at29['2'], output['m'])

tsk_ctrl = ctrl.ControlSystem([rule1, rule2])

tsk = ctrl.ControlSystemSimulation(tsk_ctrl)

i = 8
tsk.input['at0'] = network.input_test[i][0]
tsk.input['at1'] = network.input_test[i][1]
tsk.input['at2'] = network.input_test[i][2]
tsk.input['at3'] = network.input_test[i][3]
tsk.input['at4'] = network.input_test[i][4]
tsk.input['at5'] = network.input_test[i][5]
tsk.input['at6'] = network.input_test[i][6]
tsk.input['at7'] = network.input_test[i][7]
tsk.input['at8'] = network.input_test[i][8]
tsk.input['at9'] = network.input_test[i][9]
tsk.input['at10'] = network.input_test[i][10]
tsk.input['at11'] = network.input_test[i][11]
tsk.input['at12'] = network.input_test[i][12]
tsk.input['at13'] = network.input_test[i][13]
tsk.input['at14'] = network.input_test[i][14]
tsk.input['at15'] = network.input_test[i][15]
tsk.input['at16'] = network.input_test[i][16]
tsk.input['at17'] = network.input_test[i][17]
tsk.input['at18'] = network.input_test[i][18]
tsk.input['at19'] = network.input_test[i][19]
tsk.input['at20'] = network.input_test[i][20]
tsk.input['at21'] = network.input_test[i][21]
tsk.input['at22'] = network.input_test[i][22]
tsk.input['at23'] = network.input_test[i][23]
tsk.input['at24'] = network.input_test[i][24]
tsk.input['at25'] = network.input_test[i][25]
tsk.input['at26'] = network.input_test[i][26]
tsk.input['at27'] = network.input_test[i][27]
tsk.input['at28'] = network.input_test[i][28]
tsk.input['at29'] = network.input_test[i][29]

# Crunch the numbers
tsk.compute()

print(tsk.output['output'])
print(network.output_test[i])
output.view(sim=tsk)