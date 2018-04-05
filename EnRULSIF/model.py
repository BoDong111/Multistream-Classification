import math, sklearn.metrics.pairwise as sk
from sklearn import svm
import numpy as np
import random, sys


class Model(object):
    def __init__(self):
        self.model = None
        self.weight = 0.0

    def test(self, testdatax):
        confidences = []
        count = 0.0
        if len(testdatax)==1:
            testdata = np.reshape(testdatax, (1, -1))
        predictions = self.model.predict(testdatax)
        print("predictions",predictions)
        probs = self.model.predict_proba(testdatax)
        for i in range(0, len(testdatax)):
            for j in range(len(self.model.classes_)):
                print("self.model.classes_", self.model.classes_)
                if self.model.classes_[j] == predictions[i]:
                    confidences.append(probs[i][j])
                    #count +=1
                    break
        #print "count", count
        return predictions, confidences

    def computeModelWeightRULSIF(self, data):
        totConf = 0.0
        predictedClass, confidences = self.test(data)
        for i in range(0, len(confidences)):
            totConf += confidences[i]
        return totConf/len(data)

    def __computeWeight(self, errorRate):
        if errorRate <=0.5:
            if errorRate ==0:
                errorRate ==0.01
            return 0.5*math.log((1-errorRate)/errorRate)
        else:
            return 0.01



