import numpy as np
from ddapred import DDAPRED
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from graph_utils import Gauss_profile,impute_zeros
from SNF import SNF

#load data
intMat=np.loadtxt('./data/drug_dis_mat.txt')
W1=np.loadtxt('./data/Sim_target_go_mat.txt')
W2=np.loadtxt('./data/Sim_target_domain_mat.txt')
W3=np.loadtxt('./data/Sim_pubchem_mat.txt')
drugMat1=SNF([W1,W2,W3],20,20)
diseaseMat1=np.loadtxt('./data/Sim_disease.txt')
drugMat=SNF([W1,W2,W3,Gauss_profile(intMat)],20,20)
diseaseMat=SNF([diseaseMat1,Gauss_profile(intMat.T)],20,20)
test_label=intMat.flatten()
score=np.zeros(intMat.shape[0]*intMat.shape[1])

#10-fold cross validation
d1,t1=np.where(intMat==1)
d2,t2=np.where(intMat==0)
prng=np.random.RandomState(17)
fn=10
fold1=prng.permutation(len(d1))%fn
fold2=prng.permutation(len(d2))%fn

for f in range(fn):
    intMat1=intMat.copy()
    intMat1[d1[fold1==f],t1[fold1==f]]=0
    model1=DDAPRED()
    dr_Gauss1=Gauss_profile(impute_zeros(intMat1,drugMat1))
    di_Gauss1=Gauss_profile(impute_zeros(intMat1.T,diseaseMat1))
    dr_snf=SNF([W1,W2,W3,dr_Gauss1],20,20)
    di_snf=SNF([diseaseMat1,di_Gauss1],20,20)
    model1.fix_model(intMat1,dr_snf,di_snf,22)
    rows=np.array(list(zip(np.hstack((d1[fold1==f],d2[fold2==f])),np.hstack((t1[fold1==f],t2[fold2==f])))))
    score1=model1.predict_scores(rows)
    score[rows[:,0]*intMat1.shape[1]+rows[:,1]]=score1
    
prec, rec, thr = precision_recall_curve(test_label, np.array(score))
aupr_val = auc(rec, prec)
print(aupr_val)
fpr, tpr, thr = roc_curve(test_label, np.array(score))
auc_val = auc(fpr, tpr)
print(auc_val)
