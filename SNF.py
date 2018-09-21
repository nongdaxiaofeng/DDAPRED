import numpy as np


def FindDominantSet(W,K):
	m,n = W.shape
	DS = np.zeros((m,n))
	for i in range(m):
		index =  np.argsort(W[i,:])[-K:] # get the closest K neighbors 
		DS[i,index] = W[i,index] # keep only the nearest neighbors 

	#normalize by sum 
	B = np.sum(DS,axis=1)
	B = B.reshape(len(B),1)
	DS = DS/B
	return DS


def normalized(W,ALPHA):
    m,n = W.shape
    DS=W-np.diag(np.diag(W))
    B=np.sum(DS,1)
    p=B!=0
    B=B.reshape(len(B),1)
    DS[p,:]=DS[p,:]/B[p,:]
    DS = DS+ALPHA*np.identity(m)
    return (DS+np.transpose(DS))/2


def SNF(Wall,K,t,ALPHA=1):
    C = len(Wall)
    m,n = Wall[0].shape

    for i in range(C):
        B = np.sum(Wall[i],axis=1)
        len_b = len(B)
        B = B.reshape(len_b,1)
        Wall[i] = Wall[i]/B
        Wall[i] = (Wall[i]+np.transpose(Wall[i]))/2

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i],K))
            
    for iteration in range(t):
        Wsum = np.zeros((m,n))
        for i in range(C):
            Wall[i] = normalized(Wall[i],ALPHA)
            Wsum += Wall[i]
        for i in range(C):
            Wall[i] = np.dot(np.dot(newW[i], (Wsum - Wall[i])),np.transpose(newW[i]))/(C-1)
       
    Wsum = np.zeros((m,n))
    for i in range (C):
        Wall[i] = normalized(Wall[i],ALPHA)
        Wsum+=Wall[i]

    W = Wsum/C
    W=normalized(W,ALPHA)
    return W
