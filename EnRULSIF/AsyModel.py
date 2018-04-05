import math, numpy as np
from random import randint
import random
import csv
from Classification import Classification
import time
from Ensemble import Ensemble
from pelts import pelt
from costs import normal_mean

class Update(object):

    def __init__(self):
        self.Ensemble=Ensemble(3)
        super(Update, self).__init__()

    @staticmethod
    def readdata(sourcex_matrix=None, sourcey_matrix=None,targetx_matrix=None, targety_matrix=None,src_path='datasets/syndata_002_normalized_no_novel_class_source_stream.csv',
                   tgt_path='datasets/syndata_002_normalized_no_novel_class_target_stream.csv', src_size=None, tgt_size=None):
        """ 
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        if sourcex_matrix is None:
            sourcex_matrix_, sourcey_matrix = Classification.read_csv(src_path, None)   # matrix_ is source data
        else:
            sourcex_matrix_ = sourcex_matrix
            sourcey_matrix_ = sourcey_matrix
        matrix_ = sourcex_matrix_[:src_size, :]

        if targetx_matrix is None:
            targetx_ ,targety_= Classification.read_csv(tgt_path, size=None)
        else:
            targetx_ = targetx_matrix
            targety_ = targety_matrix
        labellist = []
        for i in range(0, len(targety_)):
            if targety_[i] not in labellist:
                labellist.append(targety_[i])
        sourcey_label = []
        for i in range(0, len(sourcey_matrix)):
            sourcey_label.append(labellist.index(sourcey_matrix[i]))

        for i in range(0, len(targety_)):
            if targety_[i] not in labellist:
                labellist.append(targety_[i])
        targety_label = []
        for i in range(0, len(targety_)):
            targety_label.append(labellist.index(targety_[i]))
        return sourcex_matrix_,sourcey_label, targetx_, targety_label

    def Process(self, sourcex,sourcey, targetx,targety,subsize):
        # fixed size windows for source stream and target stream

        sourceIndex = 0
        targetIndex = 0
        src_count = 0
        tgtchange_count = 0
        threshold = 1.0
        src_size, _ = sourcex.shape
        tgt_size, _ = targetx.shape
        #true_label = []
        #for i in range(len(np.array(targety))):
            #if np.array(targety)[i] == 'class1':
                #true_label.append(1)
            #if np.array(targety)[i] == 'class2':
                #true_label.append(2)
            #if np.array(targety)[i] == 'class3':
                #true_label.append(3)
            #if np.array(targety)[i] == 'class4':
                #true_label.append(4)
            #if np.array(targety)[i] == 'class5':
                #true_label.append(5)
            #if np.array(targety)[i] == 'class6':
                #true_label.append(6)
            #if np.array(targety)[i] == 'class7':
                #true_label.append(7)

        windowsize = 1000
        sourcewindowstart = 0
        sourcewindowend = sourcewindowstart + windowsize -1
        targetwindowstart = 0
        targetwindowend = targetwindowstart + windowsize - 1
        sourcexwindow = sourcex[sourcewindowstart:sourcewindowend]
        sourceywindow = sourcey[sourcewindowstart:sourcewindowend]
        targetxwindow = targetx[targetwindowstart:targetwindowend]
        targetywindow = targety[targetwindowstart:targetwindowend]

        ### get the initial model by using the first source and target windows
        alpha = 0.05
        b = targetxwindow.T.shape[1];
        fold = 5
        sigma_list = Classification.sigma_list(np.array(targetxwindow.T),
                                               np.array(sourcexwindow.T));
        lambda_list = Classification.lambda_list();
        srcx_array = np.array(sourcexwindow.T);
        trgx_array = np.array(targetxwindow.T);
        (thetah_old, w, sce_old, sigma_old) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b, fold)

        self.Ensemble.generateNewModelRULSIF(targetxwindow, sourcexwindow, sourceywindow, alpha, sigma_list,
                                             lambda_list, b, fold,subsize)
        # print "update model", src_size, source.shape
        truelablecount = 0.0
        totalcount = 0.0


        #tmpsrccount = 0
        tmptrgcount = 0
        changeindex = -1
        updatestartindex = 0
        while True:
            if sourcewindowend >= src_size or targetwindowend >= tgt_size:
                break

            data_type = randint(1, 10)
            if data_type < 2:
                print("get data from source")
                sourcewindowstart+=1
                sourcewindowend+=1
                sourcexwindow = sourcex[sourcewindowstart:sourcewindowend]
                sourceywindow = sourcey[sourcewindowstart:sourcewindowend]
                sourceIndex += 1
                #src_count += 1
                #tmpsrccount += 1
                print("sourceIndex", sourceIndex)
            else:
                print("get data from target")
                targetwindowstart+=1
                targetwindowend+=1
                targetxwindow = targetx[targetwindowstart:targetwindowend]
                targetywindow = targety[targetwindowstart:targetwindowend]
                targetIndex += 1
                tgtchange_count += 1
                tmptrgcount += 1
                print("targetIndex", targetIndex)
            if tgtchange_count>=1000:
                changeindex = 1
                tgtchange_count = 0
                confidencelist = []
                for i in range(targetwindowstart, targetwindowend+1):
                    instanceresult = self.Ensemble.evaluateEnsembleRULSIF(targetx[i])
                    confidencelist.append(instanceresult[1])
                confvar = np.var(confidencelist)
                changetestresult = pelt(normal_mean(confidencelist, confvar), len(confidencelist))
                if len(changetestresult)>1:
                    alpha = 0.05
                    b = targetxwindow.T.shape[1];
                    fold = 5
                    sigma_list = Classification.sigma_list(np.array(targetxwindow.T),
                                                           np.array(sourcexwindow.T));
                    lambda_list = Classification.lambda_list();
                    self.Ensemble.generateNewModelRULSIF(targetxwindow, sourcexwindow, sourceywindow, alpha, sigma_list,
                                                         lambda_list, b, fold, subsize)

                #x_nu = np.array(targetxwindow.T);
                #(thetah_new, w, sce_new, sigma_new) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list,
                                                                             #lambda_list, b, fold)
                #targetweight_old = Classification.compute_target_weight(thetah_old, sce_old, sigma_old, x_nu)
                #targetweight_new = Classification.compute_target_weight(thetah_new, sce_new, sigma_new, x_nu)
                #l_ratios = targetweight_new / targetweight_old

                #lnWeightTrgData = np.log(l_ratios, dtype='float64')
                #changeScore = np.sum(lnWeightTrgData, dtype='float64')
                #tgtchange_count=0
                #print "changeScore", changeScore
                #if changeScore > threshold:
                    #alpha = 0.05
                    #b = targetxwindow.T.shape[1];
                    #fold = 5
                    #sigma_list = Classification.sigma_list(np.array(targetxwindow.T),
                                                           #np.array(sourcexwindow.T));
                    #lambda_list = Classification.lambda_list();
                    #self.Ensemble.generateNewModelRULSIF(targetxwindow, sourcexwindow, sourceywindow, alpha, sigma_list,
                                                         #lambda_list, b, fold, subsize)



            if tmptrgcount>=2000:
                # force update model
                tmptrgcount=0
                #update predictions for updatestartindex to targetIndex
                for i in range(updatestartindex,targetIndex+1):
                    print("targetx[i]", targetx[i])
                    instanceresult = self.Ensemble.evaluateEnsembleRULSIF(targetx[i])
                    print("instanceresult", instanceresult)
                    print("instanceresult[0]", instanceresult[0])
                    print("truelabel[i]", targety[i])
                    if instanceresult[0] == targety[i]:
                        truelablecount +=1.0
                    totalcount +=1.0
                print("truelablecount",truelablecount)
                print("totalcount", totalcount)
                with open('errorsyn002405.csv', 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([targetIndex, truelablecount,totalcount,truelablecount/totalcount ])
                updatestartindex = targetIndex+1
                alpha = 0.05
                b = targetxwindow.T.shape[1];
                fold = 5
                sigma_list = Classification.sigma_list(np.array(targetxwindow.T),
                                                       np.array(sourcexwindow.T));
                lambda_list = Classification.lambda_list();
                self.Ensemble.generateNewModelRULSIF(targetxwindow, sourcexwindow, sourceywindow, alpha, sigma_list,
                                                     lambda_list, b, fold,subsize)
                # print "update model", src_size, source.shape



        #restarget = self.Ensemble.evaluateEnsembleRULSIF(targetx_[0])
        #print "restarget", restarget
        #print "modelclass", self.model.classes_
        #return restarget



sourcex_matrix_,sourcey_matrix, targetx_, targety_ = Update.readdata(src_size=None)
print("done reading data")
start_time = time.clock()
up = Update()
up.Process(sourcex_matrix_,sourcey_matrix, targetx_, targety_,2)
#resTarget = ensemble.evaluateEnsembleRULSIF(targetx_[0])
#print "resTarget", resTarget
#true_y = Classification.g_true_label(targety_)
#truetrg_labellist = []

#error = Classification.get_prediction_error(updateYhat,truetrg_labellist)
end_time = time.clock()

print("Execution time for %d iterations is: %s min" % (
1000, (end_time-start_time)/60.0))
#print "update y hat pm2.5", updateYhat
print("errorsyn002trgfirst500 0404 0.05")