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
network = RBFNetwork(3,scaledData,label)
network.train()
network.test()

#centers
ci = np.asarray(network.protos)
sigma = network.spread

countProtos, attributes = ci.shape
boundary = np.zeros(shape = (countProtos, attributes, 2))

for count in range(countProtos):
    for i in range(attributes):
        xd = ci[count][i] - sigma + (2.45*sigma)/5
        xd = 0 if xd < 0 else xd
        xh = ci[count][i] + sigma - (2.45*sigma)/5
        xh = 0 if xh < 0 else xh
        boundary[count][i] = [round(xd, 4), round(xh, 4)]

# fuzzy

# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl
#
# at0 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at0')
# for i in range(6):
#     at0[str(i)] = fuzz.trapmf(at0.universe,
#                               [boundary[i][0][0], boundary[i][0][0], boundary[i][0][1], boundary[i][0][1]])
#
# at1 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at1')
# for i in range(6):
#     at1[str(i)] = fuzz.trapmf(at1.universe,
#                               [boundary[i][1][0], boundary[i][1][0], boundary[i][1][1], boundary[i][1][1]])
#
# at2 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at2')
# for i in range(6):
#     at2[str(i)] = fuzz.trapmf(at2.universe,
#                               [boundary[i][2][0], boundary[i][2][0], boundary[i][2][1], boundary[i][2][1]])
#
# at3 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at3')
# for i in range(6):
#     at3[str(i)] = fuzz.trapmf(at3.universe,
#                               [boundary[i][3][0], boundary[i][3][0], boundary[i][3][1], boundary[i][3][1]])
#
# at4 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at4')
# for i in range(6):
#     at4[str(i)] = fuzz.trapmf(at4.universe,
#                               [boundary[i][4][0], boundary[i][4][0], boundary[i][4][1], boundary[i][4][1]])
#
# at5 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at5')
# for i in range(6):
#     at5[str(i)] = fuzz.trapmf(at5.universe,
#                               [boundary[i][5][0], boundary[i][5][0], boundary[i][5][1], boundary[i][5][1]])
#
# at6 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at6')
# for i in range(6):
#     at6[str(i)] = fuzz.trapmf(at6.universe,
#                               [boundary[i][6][0], boundary[i][6][0], boundary[i][6][1], boundary[i][6][1]])
#
# at7 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at7')
# for i in range(6):
#     at7[str(i)] = fuzz.trapmf(at7.universe,
#                               [boundary[i][7][0], boundary[i][7][0], boundary[i][7][1], boundary[i][7][1]])
#
# at8 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at8')
# for i in range(6):
#     at8[str(i)] = fuzz.trapmf(at8.universe,
#                               [boundary[i][8][0], boundary[i][8][0], boundary[i][8][1], boundary[i][8][1]])
#
# at9 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at9')
# for i in range(6):
#     at9[str(i)] = fuzz.trapmf(at9.universe,
#                               [boundary[i][9][0], boundary[i][9][0], boundary[i][9][1], boundary[i][9][1]])
#
# at10 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at10')
# for i in range(6):
#     at10[str(i)] = fuzz.trapmf(at10.universe,
#                                [boundary[i][10][0], boundary[i][10][0], boundary[i][10][1], boundary[i][10][1]])
#
# at11 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at11')
# for i in range(6):
#     at11[str(i)] = fuzz.trapmf(at11.universe,
#                                [boundary[i][11][0], boundary[i][11][0], boundary[i][11][1], boundary[i][11][1]])
#
# at12 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at12')
# for i in range(6):
#     at12[str(i)] = fuzz.trapmf(at12.universe,
#                                [boundary[i][12][0], boundary[i][12][0], boundary[i][12][1], boundary[i][12][1]])
#
# at13 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at13')
# for i in range(6):
#     at13[str(i)] = fuzz.trapmf(at13.universe,
#                                [boundary[i][13][0], boundary[i][13][0], boundary[i][13][1], boundary[i][13][1]])
#
# at14 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at14')
# for i in range(6):
#     at14[str(i)] = fuzz.trapmf(at14.universe,
#                                [boundary[i][14][0], boundary[i][14][0], boundary[i][14][1], boundary[i][14][1]])
#
# at15 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at15')
# for i in range(6):
#     at15[str(i)] = fuzz.trapmf(at15.universe,
#                                [boundary[i][15][0], boundary[i][15][0], boundary[i][15][1], boundary[i][15][1]])
#
# at16 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at16')
# for i in range(6):
#     at16[str(i)] = fuzz.trapmf(at16.universe,
#                                [boundary[i][16][0], boundary[i][16][0], boundary[i][16][1], boundary[i][16][1]])
#
# at17 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at17')
# for i in range(6):
#     at17[str(i)] = fuzz.trapmf(at17.universe,
#                                [boundary[i][17][0], boundary[i][17][0], boundary[i][17][1], boundary[i][17][1]])
#
# at18 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at18')
# for i in range(6):
#     at18[str(i)] = fuzz.trapmf(at18.universe,
#                                [boundary[i][18][0], boundary[i][18][0], boundary[i][18][1], boundary[i][18][1]])
#
# at19 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at19')
# for i in range(6):
#     at19[str(i)] = fuzz.trapmf(at19.universe,
#                                [boundary[i][19][0], boundary[i][19][0], boundary[i][19][1], boundary[i][19][1]])
#
# at20 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at20')
# for i in range(6):
#     at20[str(i)] = fuzz.trapmf(at20.universe,
#                                [boundary[i][20][0], boundary[i][20][0], boundary[i][20][1], boundary[i][20][1]])
#
# at21 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at21')
# for i in range(6):
#     at21[str(i)] = fuzz.trapmf(at21.universe,
#                                [boundary[i][21][0], boundary[i][21][0], boundary[i][21][1], boundary[i][21][1]])
#
# at22 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at22')
# for i in range(6):
#     at22[str(i)] = fuzz.trapmf(at22.universe,
#                                [boundary[i][22][0], boundary[i][22][0], boundary[i][22][1], boundary[i][22][1]])
#
# at23 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at23')
# for i in range(6):
#     at23[str(i)] = fuzz.trapmf(at23.universe,
#                                [boundary[i][23][0], boundary[i][23][0], boundary[i][23][1], boundary[i][23][1]])
#
# at24 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at24')
# for i in range(6):
#     at24[str(i)] = fuzz.trapmf(at24.universe,
#                                [boundary[i][24][0], boundary[i][24][0], boundary[i][24][1], boundary[i][24][1]])
#
# at25 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at25')
# for i in range(6):
#     at25[str(i)] = fuzz.trapmf(at25.universe,
#                                [boundary[i][25][0], boundary[i][25][0], boundary[i][25][1], boundary[i][25][1]])
#
# at26 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at26')
# for i in range(6):
#     at26[str(i)] = fuzz.trapmf(at26.universe,
#                                [boundary[i][26][0], boundary[i][26][0], boundary[i][26][1], boundary[i][26][1]])
#
# at27 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at27')
# for i in range(6):
#     at27[str(i)] = fuzz.trapmf(at27.universe,
#                                [boundary[i][27][0], boundary[i][27][0], boundary[i][27][1], boundary[i][27][1]])
#
# at28 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at28')
# for i in range(6):
#     at28[str(i)] = fuzz.trapmf(at28.universe,
#                                [boundary[i][28][0], boundary[i][28][0], boundary[i][28][1], boundary[i][28][1]])
#
# at29 = ctrl.Antecedent(np.arange(0, 1, 0.0001), 'at29')
# for i in range(6):
#     at29[str(i)] = fuzz.trapmf(at29.universe,
#                                [boundary[i][29][0], boundary[i][29][0], boundary[i][29][1], boundary[i][29][1]])
#
# output = ctrl.Consequent(np.arange(0, 1, 0.01), 'output')
# output['m'] = fuzz.trapmf(output.universe, [0, 0, 0.5, 0.5])
# # output['idk'] = fuzz.trimf(output.universe, [0, 0.5, 1])
# output['b'] = fuzz.trapmf(output.universe, [0.5, 0.5, 1, 1])
#
# rule1 = ctrl.Rule(at0['0'] | at1['0'] | at2['0'] | at3['0'] | at4['0'] | at5['0'] | at6['0'] | at7['0'] | at8['0'] | at9['0'] | at10['0'] | at11['0'] | at12['0'] | at13['0'] | at14['0'] | at15['0'] | at16['0'] | at17['0'] | at18['0'] | at19['0'] | at20['0'] | at21['0'] | at22['0'] | at23['0'] | at24['0'] | at25['0'] | at26['0'] | at27['0'] | at28['0'] | at29['0'], output['m'])
# rule2 = ctrl.Rule(at0['1'] | at1['1'] | at2['1'] | at3['1'] | at4['1'] | at5['1'] | at6['1'] | at7['1'] | at8['1'] | at9['1'] | at10['1'] | at11['1'] | at12['1'] | at13['1'] | at14['1'] | at15['1'] | at16['1'] | at17['1'] | at18['1'] | at19['1'] | at20['1'] | at21['1'] | at22['1'] | at23['1'] | at24['1'] | at25['1'] | at26['1'] | at27['1'] | at28['1'] | at29['1'], output['m'])
# rule3 = ctrl.Rule(at0['2'] | at1['2'] | at2['2'] | at3['2'] | at4['2'] | at5['2'] | at6['2'] | at7['2'] | at8['2'] | at9['2'] | at10['2'] | at11['2'] | at12['2'] | at13['2'] | at14['2'] | at15['2'] | at16['2'] | at17['2'] | at18['2'] | at19['2'] | at20['2'] | at21['2'] | at22['2'] | at23['2'] | at24['2'] | at25['2'] | at26['2'] | at27['3'] | at28['2'] | at29['2'], output['m'])
# rule4 = ctrl.Rule(at0['3'] | at1['3'] | at2['3'] | at3['3'] | at4['3'] | at5['3'] | at6['3'] | at7['3'] | at8['3'] | at9['3'] | at10['3'] | at11['3'] | at12['3'] | at13['3'] | at14['3'] | at15['3'] | at16['3'] | at17['3'] | at18['3'] | at19['3'] | at20['3'] | at21['3'] | at22['3'] | at23['3'] | at24['3'] | at25['3'] | at26['3'] | at27['3'] | at28['3'] | at29['3'], output['b'])
# rule5 = ctrl.Rule(at0['4'] | at1['4'] | at2['4'] | at3['4'] | at4['4'] | at5['4'] | at6['4'] | at7['4'] | at8['4'] | at9['4'] | at10['4'] | at11['4'] | at12['4'] | at13['4'] | at14['4'] | at15['4'] | at16['4'] | at17['4'] | at18['4'] | at19['4'] | at20['4'] | at21['4'] | at22['4'] | at23['4'] | at24['4'] | at25['4'] | at26['4'] | at27['4'] | at28['4'] | at29['4'], output['b'])
# rule6 = ctrl.Rule(at0['5'] | at1['5'] | at2['5'] | at3['5'] | at4['5'] | at5['5'] | at6['5'] | at7['5'] | at8['5'] | at9['5'] | at10['5'] | at11['5'] | at12['5'] | at13['5'] | at14['5'] | at15['5'] | at16['5'] | at17['5'] | at18['5'] | at19['5'] | at20['5'] | at21['5'] | at22['5'] | at23['5'] | at24['5'] | at25['5'] | at26['5'] | at27['5'] | at28['5'] | at29['5'], output['b'])
#
# tsk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
#
# tsk = ctrl.ControlSystemSimulation(tsk_ctrl)
#
# i = 100
# tsk.input['at0'] = network.input_test[i][0]
# tsk.input['at1'] = network.input_test[i][1]
# tsk.input['at2'] = network.input_test[i][2]
# tsk.input['at3'] = network.input_test[i][3]
# tsk.input['at4'] = network.input_test[i][4]
# tsk.input['at5'] = network.input_test[i][5]
# tsk.input['at6'] = network.input_test[i][6]
# tsk.input['at7'] = network.input_test[i][7]
# tsk.input['at8'] = network.input_test[i][8]
# tsk.input['at9'] = network.input_test[i][9]
# tsk.input['at10'] = network.input_test[i][10]
# tsk.input['at11'] = network.input_test[i][11]
# tsk.input['at12'] = network.input_test[i][12]
# tsk.input['at13'] = network.input_test[i][13]
# tsk.input['at14'] = network.input_test[i][14]
# tsk.input['at15'] = network.input_test[i][15]
# tsk.input['at16'] = network.input_test[i][16]
# tsk.input['at17'] = network.input_test[i][17]
# tsk.input['at18'] = network.input_test[i][18]
# tsk.input['at19'] = network.input_test[i][19]
# tsk.input['at20'] = network.input_test[i][20]
# tsk.input['at21'] = network.input_test[i][21]
# tsk.input['at22'] = network.input_test[i][22]
# tsk.input['at23'] = network.input_test[i][23]
# tsk.input['at24'] = network.input_test[i][24]
# tsk.input['at25'] = network.input_test[i][25]
# tsk.input['at26'] = network.input_test[i][26]
# tsk.input['at27'] = network.input_test[i][27]
# tsk.input['at28'] = network.input_test[i][28]
# tsk.input['at29'] = network.input_test[i][29]
#
# # Crunch the numbers
# tsk.compute()
#
# print(tsk.output['output'])
# print(network.output_test[i])
# output.view(sim=tsk)