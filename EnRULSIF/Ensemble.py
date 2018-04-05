import math
from Classification import Classification
from model import Model

class Ensemble(object):

    def __init__(self, ensemble_size):
        self.models = []
        self.size = 3

    def updateWeight(self, data, isSource):
        for m in self.models:
            m.computeModelWeight(data, isSource)

    def reEvalModelWeights(self, data):
        for i in range(0, len(self.models)):
            self.models[i].weight = self.models[i].computeModelWeightRULSIF(data)

    def __addModelRULSIF(self, model, data):
        index = 0
        self.reEvalModelWeights(data)
        if len(self.models) <self.size:
            self.models.append(model)
            index = len(self.models)-1
        else:
            index = self.__getLeastDesirableModelRULSIF()
            if self.models[index].weight < model.weight:
                print('Least desirable model removed at ' + str(index))
                self.models[index] = model
            else:
                print('New model was not added as its weight is less than all of the existing models')
                return -1
        return index

    def __getLeastDesirableModelRULSIF(self):
        weights = {}
        for i in xrange(len(self.models)):
            weights[i] = self.models[i].weight

        keys = sorted(weights, key=weights.get)

        return keys[0]

    def generateNewModelRULSIF(self, trgx_matrix, srcx_matrix, srcy_matrix, alpha, sigma_list, lambda_list, b, fold, subsize):
        model = Model()

        if len(srcx_matrix) == 0 or len(trgx_matrix) == 0:
            raise Exception('Source or Target stream should have some elements')

        # Create new model
        print('Target model creation')
        model.model = Classification.get_model(trgx_matrix, srcx_matrix, srcy_matrix, alpha, sigma_list, lambda_list, b,
                                               fold,subsize)

        # compute source and target weight
        print('Computing model weights')
        model.weight = model.computeModelWeightRULSIF(trgx_matrix)

        # update ensemble
        index = self.__addModelRULSIF(model, trgx_matrix)
        if index != -1:
            print('Ensemble updated at ' + str(index))




    def evaluateEnsembleRULSIF(self, dataInstance):
        confSum = {}
        weightSum = {}
        for m in self.models:
            # test data instance in each model
            predictedClass, confidence = m.test(dataInstance)
        # gather result
            if predictedClass[0] in confSum:
                confSum[predictedClass[0]] += confidence[0]
                weightSum[predictedClass[0]] += m.weight
            else:
                confSum[predictedClass[0]] = confidence[0]
                weightSum[predictedClass[0]] = m.weight

    # get maximum voted class label
        classMax = 0.0
        sorted(confSum, key=confSum.get, reverse=True)
        classMax = sorted(confSum, key=confSum.get, reverse=True)[0]
        #classMax = confSum.keys()[0]

        return [classMax, confSum[classMax] / len(self.models)]


"""
Get summary of models in ense

"""


#def getEnsembleSummary(self):
#    summry = '************************* E N S E M B L E    S U M M A R Y ************************\n'
#    summry += 'Ensemble has currently ' + str(len(self.models)) + ' models.\n'
#    for i in range(len(self.models)):
#        summry += 'Model' + str(i + 1) + ': weights<' + str(self.models[i].weight) + '>\n'
#    return summry