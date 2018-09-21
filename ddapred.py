import numpy as np


class DDAPRED:
    def __init__(self, K1=5, r=25, theta=1.0, c1=0.1,c2=30,  max_iter=200):
        self.K1 = int(K1)
        self.num_factors = int(r)
        self.theta = float(theta)
        self.lambda_d = float(c1)
        self.lambda_t = float(c1)
        self.alpha = float(c2)
        self.beta = float(c2)
        self.max_iter = int(max_iter)

    def AGD_optimization(self, seed=None):
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_diseases, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_diseases, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_diseases, self.V.shape[1]))
        last_log = self.log_likelihood()
        for t in range(self.max_iter):
            dg = self.deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self.deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self.log_likelihood()
            delta_log = (curr_log-last_log)/abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log
        #print(delta_log)

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A[A>0]=1.0/(1+np.exp(-A[A>0]))
        A[A<0]=np.exp(A[A<0])/(1+np.exp(A[A<0]))
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        return vec_deriv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik = np.sum(B)
        A[A>0] += np.log(1+np.exp(-A[A>0]))
        A[A<0]=np.log(1+np.exp(A[A<0]))
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, diseaseMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = diseaseMat - np.diag(np.diag(diseaseMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, intMat, drugMat, diseaseMat, seed=None):
        self.num_drugs, self.num_diseases = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_diseases))
        self.intMat = intMat
        self.intMat1 = self.ones
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_diseases = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, diseaseMat)
        self.AGD_optimization(seed)


    def predict_scores(self, test_data):
        scores = []
        for d, t in test_data:
            val = np.sum(self.U[d, :]*self.V[t, :])
            if val>=0:
                scores.append(1/(1+np.exp(-val)))
            else:
                scores.append(np.exp(val)/(1+np.exp(val)))
        return np.array(scores)
