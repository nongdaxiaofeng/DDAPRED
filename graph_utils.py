import numpy as np

    
def Gauss_profile(adj):
	bw = 1
	ga = np.dot(adj,np.transpose(adj))
	ga = bw*ga/np.mean(np.diag(ga))
	di = np.diag(ga)
	x =  np.tile(di,(len(di),1))
	d =x+np.transpose(x)-2*ga
	return np.exp(-d)

def impute_zeros(inMat,inSim,k=5):
    mat=inMat.copy()
    sim=inSim.copy()
    indexZero = np.where(~mat.any(axis=1))[0]
    np.fill_diagonal(sim,0)
    if len(indexZero) > 0:
        sim[:,indexZero] = 0
    for i in indexZero:
        currSimForZeros = sim[i,:]
        indexRank = np.argsort(currSimForZeros)
        indexNeig = indexRank[-k:]
        simCurr = currSimForZeros[indexNeig]
        mat_known = mat[indexNeig, :]
        if sum(simCurr) >0:
            mat[i,: ] = np.dot(simCurr ,mat_known) / sum(simCurr)
            
    return mat