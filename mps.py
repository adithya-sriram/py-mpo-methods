import numpy as np
import copy
from tensor import *

class MPS:
    
    #An MPS object has two attributes: 1) L is the system size in 1D. 2) W is the list of tensor objects.
    def __init__(self, W):
        self.L = len(W)
        self.Tensors = W
        self.norm = 1

    def __getitem__(self, tensor_index):
          return self._floors[floor_number]
    
    # Create an MPO with a set of predefined tensors. This method will check to see if your tensors are properly formatted. 
    # It will check if the 1st tensor has shape (1, n, m,...), if all the link dimensions match up, and if the last tensor has shape (...,n, m, 1)    
    @staticmethod
    def finiteMPS(W):
        for i in range(1,len(W)):
            if W[i-1].tensor.shape[-1] != W[i].tensor.shape[0]:
                raise Exception('Link between ' + str(i-1) + ' and ' + str(i) + ' does not have matching dimensions')
            elif i != 0 and not any(x in W[i].inds for x in W[i-1].inds):
                raise Exception('Tensors ' + str(i-1) + ' and ' + str(i) + ' do not share a common index')
        return MPS(W)
    
    # This method is used when we want to construct an MPO based on a given periodic Hamiltonian (e.g. Ising model)
    @staticmethod
    def translationalMPS(L, Op):
        W = []
        for i in range(L):
            W.append(Op)

        TensorTrain = []
        left = np.zeros((1,W[0].shape[0]))
        left[0,0] = 1
        right = np.zeros((W[-1].shape[-1],1))
        right[-1,-1] = 1
        W[0] = np.tensordot(left, W[0],1)
        W[-1] = np.tensordot(W[-1], right, 1)

        a1 = [Index(x) for x in W[0].shape]
        a1[0] = Index(W[0].shape[0], 'Link l=0')
        a1[-1] = Index(W[0].shape[-1], 'Link l=1')
        TensorTrain.append(Tensor(W[0], a1))
        for i in range(1,L):
            temp = [Index(x) for x in W[i].shape]
            temp[0] = TensorTrain[i-1].inds[-1]
            temp[-1] = Index(W[i].shape[-1], 'Link l=' + str(i+1))
            TensorTrain.append(Tensor(W[i], temp))

        return MPS(TensorTrain)
    
    # Method to contract the network over the shared bonds to bring network into matrix form. WARNING: for larger systems this will crash the kernel
    def stateToVector(self):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        prod = W[0]
        for i in range(1,self.L):
            prod = np.einsum('ija,abd->ijbd',prod,W[i])
            #print(prod.shape)
            prod = prod.reshape((1,prod.shape[1]*prod.shape[2],prod.shape[3]))
        prod = prod.reshape((prod.size,1))
        return prod
    
    # Simpler way to get the matrix form of a particular tensor in the network
    def getTensor(self, i):
        return self.Tensors[i].tensor
    
    def makeLeftCanonical(self):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        W[0] = W[0].reshape((W[0].shape[1],W[0].shape[-1]))
        q,r = np.linalg.qr(W[0])
        W[0] = q.reshape((1,q.shape[0],q.shape[1]))
        W[1] = np.einsum('ij,jkl->ikl',r,W[1])


        for i in range(1, self.L-1):
            q,r = np.linalg.qr(W[i].reshape((W[i].size//2,W[i].size//2)))
            W[i] = q.reshape(W[i].shape)
            W[i+1] = np.einsum('ij,jkl->ikl',r,W[i+1])

        W[-1] = W[-1].reshape((W[-1].size,1))
        q,r = np.linalg.qr(W[-1])
        W[-1] = q.reshape(W[-1].shape)
        for i in range(len(W)):
            self.Tensors[i].tensor = W[i]
            
    def makeRightCanonical(self):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        for i in range(self.L-1,-1,-1):
            Q = W[i].reshape((W[i].shape[0], W[i].shape[1]*W[i].shape[2]))
            u,s,v = np.linalg.svd(Q,full_matrices = False)
            s_mat = np.zeros((s.size,s.size))
            for val in range(s.size):
                s_mat[val,val] = s[val]
            v = v.reshape((W[i].shape))
            if i != 0:
                W[i-1] = np.einsum('ijk,kl->ijl',W[i-1],u)
                W[i-1] = np.einsum('ijk,kl->ijl',W[i-1],s_mat)
            W[i] = v
        for i in range(self.L):
            self.Tensors[i].tensor = W[i]
            
    def compress(self, cutoff = 1e-10, maxDim = -1):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        for i in range(self.L-1,-1,-1):
            Q = W[i].reshape((W[i].shape[0], W[i].shape[1]*W[i].shape[2]))
            u,s,v = np.linalg.svd(Q,full_matrices = False)
            
            v = v.reshape((int(v.size / (2 * W[i].shape[-1])), 2, W[i].shape[-1] ))
            s_trunc = []
            for val in s:
                if val >= cutoff:
                    s_trunc.append(val)
                if maxDim != -1 and len(s_trunc) > maxDim:
                    break
            s_trunc = np.array(s_trunc)
            u = u[:,:s_trunc.shape[0]] * s_trunc
            v = v[:s_trunc.shape[0],:]
                
            
            if i != 0:
                W[i-1] = np.einsum('ijk,kl->ijl',W[i-1],u)
            W[i] = v
        for i in range(self.L):
            self.Tensors[i].tensor = W[i]
        
    def statenorm(self):
        W1 = [self.Tensors[i].tensor for i in range(self.L)]
        W2 = [self.Tensors[i].tensor for i in range(self.L)]
        prod = W1[0]
        prod = np.einsum('ijk,ajb->iakb',prod,W2[0])
        for i in range(self.L-1):
            prod = np.einsum('iakb,bmn->iakmn',prod,W2[i+1])
            prod = np.einsum('iakmn,kmp->ianp',prod,W1[i+1])
            #print(prod.shape)

        return np.linalg.norm(prod)

    def applyTwoSiteGate(self, site, T):
        W = [self.Tensors[i].tensor for i in range(self.L)]

        dL = W[site].shape[0]
        dR = W[site+1].shape[-1]
        prod = np.einsum('ijk,kbc->ijbc', W[site], W[site+1])
        prod = np.einsum('abcd,icdj->iabj',T,prod)

        prod = prod.reshape( (prod.shape[0] * prod.shape[1], prod.shape[2] * prod.shape[3] ) )
        u,s,v = np.linalg.svd(prod, full_matrices = False)
        u = u * s
        u = u.reshape( (dL, 2, int(u.size / (dL * 2)) ) )
        v = v.reshape( (u.shape[-1], 2, dR))


        self.Tensors[site].tensor = u
        self.Tensors[site+1].tensor = v
        self.compress()
        
    # Get the maximum bond dimension
    def getMaxBondDimension(self):
        maxD = 0
        W = [self.Tensors[i].tensor for i in range(self.L)]
        for i in range(len(W)):
            if maxD < max(W[i].shape):
                maxD = max(W[i].shape)
        return maxD

    # Apply single site gate   
    def applySingleSiteGate(self, site, G):
        self.Tensors[site].tensor = np.einsum('ij,ajb->aib',G,self.getTensor(site))
            
