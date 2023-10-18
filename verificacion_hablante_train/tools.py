import torch
from speechbrain.processing.PLDA_LDA import *
import numpy as np
import pickle
import h5py  as h5
from speechbrain.utils.metric_stats import EER
from tqdm import tqdm
from scipy.signal import lfilter
from scipy.signal import hilbert
from scipy.signal import firwin
from operator import itemgetter
############################ 
def readList(list_file):        
        fh    = open(list_file,'r')
        lines = fh.readlines()
        fh.close()
        feat_files = []
        id_spk     = []
        for line in lines:
            feat_file, id_s = line.split('\t')
            feat_files.append(feat_file)
            id_spk.append(int(id_s))
        
        return np.array(feat_files), np.array(id_spk)
############################
def loadFeats(root_feats,feats,feats_list):
    DATA = []
    for feat_file in tqdm(feats_list):
        file_data = root_feats + feat_file + '.h5'
        #print(file_data)
        with h5.File(file_data, "r") as f:            
            data = list(f[feats])
        DATA.append(data)    
    return np.array(DATA)
 
 
############################
def cosineValidation(enroll_data,test_data,id_clients,id_test):
    enroll_data = torch.from_numpy(enroll_data)
    test_data      = torch.from_numpy(test_data)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    threshold  = 0.9566656868281521
    NC = id_clients.size
    k=0
    #iterate across speakers
    validation = []
    print(f'[INFO] compute scores')  
    for i,sp in enumerate(id_clients):
        client = enroll_data[i,:]
        scores = cos(client, test_data)
        
        classification = scores > threshold
        ground_true    = sp == id_test
        
        results = np.vstack((scores,classification,ground_true))
        validation.append(results)
    print(f'[INFO] done')
    validation = np.hstack(validation)
    
    y_true = validation[2,:]
    y_pred = validation[1,:]
    y_scores = validation[0,:]

    positive_scores = y_scores[ y_true == 1]
    negative_scores = y_scores[ y_true != 1]
    
    total_trials = y_scores.size
    #print(f'porc: {(y_true==y_pred).sum()/y_true.size*100}\n')
    #print(f'[INFO] pos: {positive_scores.size},neg: {negative_scores.size}')
    # Final EER computation
    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    #print(f'[INFO] eer = {eer*100}')
    #min_dcf, th = minDCF(
    #    torch.tensor(positive_scores), torch.tensor(negative_scores)
    #)
    fnrs, fprs, thresholds= ComputeErrorRates(y_scores, y_true)  
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds)    
    
    return eer,mindcf, y_scores, y_true, total_trials, th
    
############################    
class whiten():
    def __init__(self):
        self.Mu     = np.array([])
        self.sigma  = np.array([])
        self.eigVals= np.array([])
        self.eigVecs= np.array([])
    ##### compute parameters
    def fit(self,X):
        self.get_sigma(X)
        
        eigVals, eigVecs = np.linalg.eig(self.sigma)
        self.eigVecs     = np.real(eigVecs)
        self.eigVals     = np.real(eigVals)
        return
    def get_center(self,X):
        if (self.Mu.size==0): 
            self.Mu =np.mean(X, axis = 0)
        X = X - self.Mu
        return X
    
    def get_sigma(self,X):
        X = self.get_center(X)
        self.sigma = np.cov(X, rowvar=False, bias=True)
        return
    #apply transform
    def transform(self,X):
        X = self.get_center(X)
        # Aplicar los vectores propios a los datos (rotar)
        X = X.dot(self.eigVecs)
        # Re-escalar los datos
        X = X / np.sqrt(self.eigVals + 1e-5)
        return  X    



def vadHilbert(x,fs,trh=0.2):
    x=x/np.max(abs(x))
    b = firwin(128, 20/(fs/2), window='hamming', pass_zero=True)
    s=hilbert(x)
    Eh = np.abs(s) #20*np.log10(np.abs(s))
    Eh = lfilter(b, 1, Eh)
    #tomar solo las partes mayores a 0.2
    umbral = trh
    vad = Eh>umbral
    
    return vad
    
    
    
    ###kaldi
def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    labels = [labels[i] for i in sorted_indexes]
    fns = []
    tns = []

    # At the end of this loop, fns[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, tns[i]
    # is the total number of times that we have correctly rejected scores
    # less than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fns.append(labels[i])
            tns.append(1 - labels[i])
        else:
            fns.append(fns[i-1] + labels[i])
            tns.append(tns[i-1] + 1 - labels[i])
    positives = sum(labels)
    negatives = len(labels) - positives

    # Now divide the false negatives by the total number of 
    # positives to obtain the false negative rates across
    # all thresholds
    fnrs = [fn / float(positives) for fn in fns]

    # Divide the true negatives by the total number of 
    # negatives to get the true negative rate. Subtract these 
    # quantities from 1 to get the false positive rates.
    fprs = [1 - tn / float(negatives) for tn in tns]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold    