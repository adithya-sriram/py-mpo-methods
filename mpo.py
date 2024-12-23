
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
import copy
from tensor import *

#An MPO (Matrix Product Object) can be a matrix product operator or matrix product state. 
class MPO:

    #An MPO object has two attributes: 1) L is the system size in 1D. 2) W is the list of tensor objects.
    def __init__(self, W):
        self.L = len(W)
        self.Tensors = W

    def __getitem__(self, tensor_index):
          return self._floors[floor_number]

    # Create an MPO with a set of predefined tensors. This method will check to see if your tensors are properly formatted. 
    # It will check if the 1st tensor has shape (1, n, m,...), if all the link dimensions match up, and if the last tensor has shape (...,n, m, 1)
    @staticmethod
    def finiteMPO(W):
        for i in range(1,len(W)):
            if W[i-1].tensor.shape[-1] != W[i].tensor.shape[0]:
                raise Exception('Link between ' + str(i-1) + ' and ' + str(i) + ' does not have matching dimensions')
            elif i != 0 and not any(x in W[i].inds for x in W[i-1].inds):
                raise Exception('Tensors ' + str(i-1) + ' and ' + str(i) + ' do not share a common index')
        return MPO(W)

    # This method is used when we want to construct an MPO based on a given periodic Hamiltonian (e.g. Ising model)
    @staticmethod
    def translationalMPO(L, Op):
        W = []
        for i in range(L):
            W.append(np.moveaxis(Op, 1, 3))

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

        return MPO(TensorTrain)

    # Method to contract the network over the shared bonds to bring network into matrix form. WARNING: for larger systems this will crash the kernel
    def toMatrixForm(self):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        
        A = np.moveaxis(W[0],3,1)
        B = np.moveaxis(W[1],3,1)

        mat = np.zeros((W[0].shape[0],W[1].shape[3],W[0].shape[1] * W[1].shape[2],W[0].shape[1] * W[1].shape[2]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                prod = np.kron(A[i,0], B[0,j])
                for a in range(1, A.shape[1]):
                    prod += np.kron(A[i,a], B[a,j])
                mat[i,j] = prod
        
        for i in range(1, self.L - 1):
            A = mat
            B = np.moveaxis(W[i+1],3,1)

            C = np.zeros((A.shape[0],B.shape[1],A.shape[2] * B.shape[3],A.shape[2] * B.shape[3]))
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    prod = np.kron(A[i,0], B[0,j])
                    for a in range(1, A.shape[1]):
                        prod += np.kron(A[i,a], B[a,j])
                    C[i,j] = prod
            mat = C
                
        mat = np.moveaxis(mat,1,3)
        
        return mat[0,:,:,0]
            

    # Simpler way to get the matrix form of a particular tensor in the network
    def getTensor(self, i):
        return self.Tensors[i].tensor
    
    # QR sweep to left canonicalize
    def makeLeftCanonical(self):
        for i in range(self.L - 1):
            if i == 0:
                V = self.Tensors[i].tensor[:,:,:,0:self.Tensors[i].tensor.shape[3]-1]
            elif i == self.L - 1:
                V = self.Tensors[i].tensor[0:self.Tensors[i].tensor.shape[0]-1,:,:,:]
            else:
                V = self.Tensors[i].tensor[0:self.Tensors[i].tensor.shape[0]-1,:,:,0:self.Tensors[i].tensor.shape[3]-1]
            V_flatten = np.zeros((V.shape[0] * V.shape[1] * V.shape[2], V.shape[3]))
            
            counter = 0
            for a in range(V.shape[0]):
                for b in range(V.shape[1]):
                    for c in range(V.shape[2]):
                        V_flatten[counter] = V[a,b,c]
                        counter += 1
            
            q,r = np.linalg.qr(V_flatten)
            factor = q[0,0]
            q = q / factor
            r = r * factor

            if i == 0:
                Q = np.zeros((1,2,2,q.shape[-1]))
            else:
                Q = np.zeros(V.shape)
            
            counter = 0
            for a in range(Q.shape[0]):
                for b in range(Q.shape[1]):
                    for c in range(Q.shape[2]):
                        Q[a,b,c] = q[counter]
                        counter += 1
            
            if Q.shape != V.shape:
                temp = np.zeros(V.shape)
                temp[:,:,:,-1] = self.Tensors[i].tensor[:,:,:,-1]
                temp[:,:,:,:-1] = np.copy(Q)
                self.Tensors[i].tensor = np.copy(temp)
                
            else:
                self.Tensors[i].tensor[0:Q.shape[0],:,:,0:Q.shape[3]] = np.copy(Q)
            
            R = np.zeros((r.shape[0] + 1, r.shape[1] + 1))
            for a in range(r.shape[0]):
                for b in range(r.shape[1]):
                    R[a,b] = r[a,b]

            R[R.shape[0] - 1, R.shape[1] - 1] = 1

            self.Tensors[i+1].tensor = np.tensordot(R, self.Tensors[i+1].tensor, 1)


    # Check if left canonical
    def isLeftCanonical(self):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        for i in range(self.L):
            W[i] = np.moveaxis(W[i],3,1)
        
        def norm(A, B):
            return np.trace(np.matmul(np.conj(A.T),B))
        
        for i in range(self.L):
            chi = W[i].shape[0] - 2
            for b in range(chi+1):
                for c in range(chi+1):
                    val = 0
                    for a in range(chi + 1):
                        val += norm(W[i][a,b], W[i][a,c])
                    if b == c and val == 0:
                        return False
                    elif b != c  and abs(val) > 1e-5:
                        return False
                return True

    # Mirror the left Canonical form to get the right canonical form
    def makeRightCanonical(self):

        self.makeLeftCanonical()
        W = np.copy([self.Tensors[i].tensor for i in range(self.L)])
        W_prime = np.copy([self.Tensors[i].tensor for i in range(self.L)])
        
        for i in range(self.L):
            W_prime[i] = np.flip(W[self.L - 1 - i].T)
            for a in range(W_prime[i].shape[0]):
                temp = np.copy(W_prime[i][a,0,0])
                W_prime[i][a,0,0] = np.copy(W_prime[i][a,1,1])
                W_prime[i][a,1,1] = np.copy(temp)
        
        for i in range(self.L):
            self.Tensors[i].tensor = W_prime[i]

    # Get the maximum bond dimension
    def getMaxBondDimension(self):
        maxD = 0
        W = [self.Tensors[i].tensor for i in range(self.L)]
        for i in range(len(W)):
            if maxD < max(W[i].shape):
                maxD = max(W[i].shape)
        return maxD
    
    # Compress the MPO
    def compress(self, cutoff = 1e-10, maxDim = -1):
        if maxDim >= 0 and maxDim < 3:
            raise Exception()
        elif self.getMaxBondDimension() <= 2:
            return
        #Right canonicalize the state
        self.makeRightCanonical()

        def dot(A, B):
            if len(A.shape) > len(B.shape):
                if A.shape[-1] != B.shape[0]:
                    raise Exception()
                
                result = np.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[1]))
                
                for a in range(A.shape[0]):
                    for b in range(A.shape[1]):
                        for c in range(A.shape[2]):
                            for d in range(B.shape[1]):
                                temp = 0
                                for e in range(A.shape[3]):
                                    temp += A[a,b,c,e] * B[e,d]
                                result[a,b,c,d] = temp
                            
            
            
            else:
                if A.shape[-1] != B.shape[0]:
                    raise Exception()
                result = np.zeros((A.shape[0], B.shape[1], B.shape[2], B.shape[3]))
                
                for a in range(A.shape[0]):
                    for b in range(B.shape[1]):
                        for c in range(B.shape[2]):
                            for d in range(B.shape[3]):
                                temp = 0
                                for e in range(A.shape[1]):
                                    temp += A[a,e] * B[e,b,c,d]
                                result[a,b,c,d] = temp
                                
            return result
        
        W = np.copy([self.Tensors[i].tensor for i in range(self.L)])
    
        for i in range(self.L - 1):
            if i == 0:
                V = W[i][:,:,:,0:W[i].shape[3]-1]
            elif i == len(W) - 1:
                V = W[i][0:W[i].shape[0]-1,:,:,:]
            else:
                V = W[i][0:W[i].shape[0]-1,:,:,0:W[i].shape[3]-1]
                
            V_flatten = np.zeros((V.shape[0] * V.shape[1] * V.shape[2], V.shape[3]))
            counter = 0
            for a in range(V.shape[0]):
                for b in range(V.shape[1]):
                    for c in range(V.shape[2]):
                        for d in range(V.shape[3]):
                            V_flatten[counter,d] = np.copy(V[a,b,c,d])
                        counter += 1

            q,r = np.linalg.qr(V_flatten)
            factor = q[0,0]
            q = q / factor
            r = r * factor

            if i == 0:
                Q = np.zeros((1,2,2,q.shape[-1]))
            else:
                Q = np.zeros(V.shape)
            
            
            counter = 0

            for a in range(Q.shape[0]):
                for b in range(Q.shape[1]):
                    for c in range(Q.shape[2]):
                        Q[a,b,c] = q[counter]
                        counter += 1
            
            if Q.shape != V.shape:
                temp = np.zeros(V.shape)
                temp[:,:,:,-1] = self.Tensors[i].tensor[:,:,:,-1]
                temp[:,:,:,:-1] = np.copy(Q)
                W[i] = np.copy(temp)    
            else:
                 W[i][0:Q.shape[0],:,:,0:Q.shape[3]] = np.copy(Q)
                    
            R = np.zeros((r.shape[0] + 1, r.shape[1] + 1))
            for a in range(r.shape[0]):
                for b in range(r.shape[1]):
                    R[a,b] = r[a,b]

            R[R.shape[0] - 1, R.shape[1] - 1] = 1
            
            Msmall = np.copy(R[1:R.shape[0]-1,1:R.shape[0] - 1])
            R_prime = np.eye(R.shape[0], R.shape[1])
            R_prime[0,1:] = np.copy(R[0,1:])

            u,s,v = np.linalg.svd(Msmall)
            
            discard = 0
            for a in range(len(s)):
                if s[a] < cutoff:
                    discard = a
                    break
            if maxDim != -1 and discard >= maxDim - 2:
                discard = maxDim - 2



            U_expanded = np.zeros((u.shape[0] + 2, u.shape[1] + 2 - discard))
            U_expanded[0,0] = 1
            U_expanded[-1,-1] = 1

            U_expanded[1:1+u.shape[0], 1:1+u.shape[1] - discard] = np.copy(u[:,:u.shape[1] - discard])

            
            W[i] = np.copy(dot(W[i], U_expanded))

            VT_expanded = np.zeros((v.shape[0] + 2 - discard, v.shape[1] + 2))
            VT_expanded[0,0] = 1
            VT_expanded[-1,-1] = 1

            for a in range(v.shape[1]):
                for b in range(v.shape[0] - discard):
                    VT_expanded[b+1,a+1] = v[b,a] * s[b]


            W[i+1] = np.copy(dot(VT_expanded @ R_prime, W[i+1]))

    
        for i in range(self.L):
            self.Tensors[i].tensor = W[i]
            self.Tensors[i].inds[0].dim = W[i].shape[0]
            self.Tensors[i].inds[-1].dim = W[i].shape[-1]


    def multiply(self, mpo2):
        W1 = np.copy([self.Tensors[i].tensor for i in range(self.L)])
        W2 = np.copy([mpo2.Tensors[i].tensor for i in range(mpo2.L)])
        
        
        for i in range(self.L):
            A = np.moveaxis(W1[i],1,3)
            B = np.moveaxis(W2[i],0,1)
            C = np.tensordot(A,B,1)
            C = np.moveaxis(C,3,1)
            C = np.moveaxis(C,3,4)
            C_partialFlatten = np.zeros((C.shape[0] * C.shape[1], C.shape[2], C.shape[3], C.shape[4], C.shape[5]))
            counter = 0
            for a in range(C.shape[0]):
                for b in range(C.shape[1]):
                    C_partialFlatten[counter,:,:,:] = C[a,b,:,:,:,:]
                    counter += 1
            C_flatten = np.zeros((C.shape[0] * C.shape[1], C.shape[2], C.shape[3], C.shape[4] * C.shape[5]))
            counter = 0
            for a in range(C.shape[4]):
                for b in range(C.shape[5]):
                    C_flatten[:,:,:,counter] = C_partialFlatten[:,:,:,a,b]
                    counter += 1
            W3 = np.copy(C_flatten)
            print(W3.shape)

            self.Tensors[i].tensor = np.copy(W3)
            self.Tensors[i].inds[0].dim = W3.shape[0]
            self.Tensors[i].inds[-1].dim = W3.shape[-1]
        
        self.compress()

    # Scalar multipy an MPO
    def scalarmult(self, scalar):
        for i in range(self.L):
            temp = np.moveaxis(self.Tensors[i].tensor,3,1)
            temp[:temp.shape[0]-1,-1] *= scalar
            temp = np.moveaxis(temp,1,3)
            self.Tensors[i].tensor = temp

    def add(self, mpo2, compress_sum = True):
        W = []
        A = np.copy(np.moveaxis(self.Tensors[0].tensor, 3,1))
        B = np.copy(np.moveaxis(mpo2.Tensors[0].tensor, 3,1))
        C = np.zeros((1, A.shape[1] - 1 + B.shape[1] - 1, A.shape[2], A.shape[3]))
        C[:,:A.shape[1] - 1] = A[0,:A.shape[1] - 1]
        C[:,A.shape[1] - 1:C.shape[1] - 1] = B[0,1:B.shape[1] - 1]
        C[0,-1] = A[0,-1] + B[0,-1]
        C = np.moveaxis(C,1,3)
        self.Tensors[0].tensor = np.copy(C)
        self.Tensors[0].inds[0].dim = C.shape[0]
        self.Tensors[0].inds[0].dim = C.shape[-1]
        
        for i in range(1,self.L-1):
            A = np.copy(np.moveaxis(self.Tensors[i].tensor, 3,1))
            B = np.copy(np.moveaxis(mpo2.Tensors[i].tensor, 3,1))
            C = np.zeros((A.shape[0] - 1 + B.shape[0] - 1, A.shape[1] - 1 + B.shape[1] - 1, A.shape[2], A.shape[3]))
            C[:A.shape[0] - 1, :A.shape[1] - 1] = A[:A.shape[0] - 1, :A.shape[1] - 1]
            C[A.shape[0] - 1:,A.shape[1] - 1:] = B[1:,1:]
            C[0, A.shape[1] - 1:C.shape[1] - 1] = B[0,1:B.shape[1] - 1]
            C[1:A.shape[0] - 1,-1] = A[1:A.shape[0] - 1,-1]
            C[0,-1] = A[0,-1] + B[0,-1]
            C = np.moveaxis(C,1,3)
            self.Tensors[i].tensor = np.copy(C)
            self.Tensors[i].inds[0].dim = C.shape[0]
            self.Tensors[i].inds[-1].dim = C.shape[-1]
        
        A = np.copy(np.moveaxis(self.Tensors[-1].tensor, 3,1))
        B = np.copy(np.moveaxis(mpo2.Tensors[-1].tensor, 3,1))
        C = np.zeros((A.shape[0] - 1 + B.shape[0] - 1, 1, A.shape[2], A.shape[3]))
        C[0,0] = A[0,-1] + B[0,-1]
        C[1:A.shape[0] - 1,0] = A[1:A.shape[0] - 1,-1]
        C[A.shape[0] - 1:,0] = B[1:,-1]
        C = np.moveaxis(C,1,3)
        self.Tensors[-1].tensor = np.copy(C)
        self.Tensors[-1].inds[0].dim = C.shape[0]
        self.Tensors[-1].inds[-1].dim = C.shape[-1]

        if compress_sum:
            self.compress()

    def applyTwoSiteGate(self, site, T):
        W = [self.Tensors[i].tensor for i in range(self.L)]
        prod = W[site]
        prod = np.einsum('ijka,abcd->ijkbcd',prod,W[site+1])
        prod = prod.reshape((1,prod.shape[1]*prod.shape[2]*prod.shape[3]*prod.shape[4],1,1))
        newstate = np.einsum('ab,ibjk->iajk',T,prod)
        q,r = np.linalg.qr(newstate)
        T1 = Tensor(q)
        T1.inds = self.Tensors[site].inds
        T2 = Tensor(r)
        T2.inds = self.Tensors[site+1].inds
        self.Tensors[site] = T1
        self.Tensors[site+1] = T2


# A JMPO is just a jax array. One must take care in using this.
class JMPO:

    # This will take an MPO object and return the equivalent tensor train in jax array form
    @staticmethod
    def toJMPO(mpo):
        W = [jnp.asarray(mpo.getTensor(i)) for i in range(mpo.L)]
        return W

    # This method is used when we want to construct an MPO based on a given periodic Hamiltonian (e.g. Ising model)
    @staticmethod
    def translationalMPO(L, Op):
        W = []
        for i in range(L):
            W.append(jnp.moveaxis(Op, 1, 3))

        left = jax.ops.index_update(jnp.zeros((1,W[0].shape[0])), jax.ops.index[0,0], 1)
        right = jax.ops.index_update(jnp.zeros((W[-1].shape[-1],1)), jax.ops.index[-1,-1], 1)
        W[0] = jnp.tensordot(left, W[0],1)
        W[-1] = jnp.tensordot(W[-1], right, 1)

        return W

    # Check if a tensor train is a valid MPO
    @staticmethod
    def isJMPO(mpo):
        for i in range(1,len(mpo)):
            if mpo[i-1].shape[-1] != mpo[i].shape[0]:
                False
        return True
    
    @staticmethod
    def makeLeftCanonical(self):
        
        W_prime = [self[i] for i in range(len(self))]
        for i in range(len(self) - 1):
            if i == 0:
                V = W_prime[i][:,:,:,0:W_prime[i].shape[3]-1]
            elif i == len(self) - 1:
                V = W_prime[i][0:W_prime[i].shape[0]-1,:,:,:]
            else:
                V = W_prime[i][0:W_prime[i].shape[0]-1,:,:,0:W_prime[i].shape[3]-1]
            V_flatten = jnp.zeros((V.shape[0] * V.shape[1] * V.shape[2], V.shape[3]))
            
            counter = 0
            for a in range(V.shape[0]):
                for b in range(V.shape[1]):
                    for c in range(V.shape[2]):
                        V_flatten = jax.ops.index_update(V_flatten, jax.ops.index[counter], V[a,b,c])
                        counter += 1
            
            q,r = jnp.linalg.qr(V_flatten)
            factor = q[0,0]
            q = q / factor
            r = r * factor
            if i == 0:
                Q = jnp.zeros((1,2,2,q.shape[-1]))
            else:
                Q = jnp.zeros(V.shape)
            
            counter = 0
            
            for a in range(Q.shape[0]):
                for b in range(Q.shape[1]):
                    for c in range(Q.shape[2]):
                        Q = jax.ops.index_update(Q, jax.ops.index[a,b,c], q[counter])
                        counter += 1
            

            if Q.shape[-1] != W_prime[i].shape[-1] - 1:
                temp = jnp.zeros(W_prime[i].shape)
                temp = jax.ops.index_update(temp, jax.ops.index[:,:,:,-1], W_prime[i][:,:,:,-1])
                temp = jax.ops.index_update(temp, jax.ops.index[:,:,:,:Q.shape[-1]], Q)
                W_prime[i] = jnp.array(temp, copy = True)
                
            else:
                W_prime[i] = jax.ops.index_update(W_prime[i], jax.ops.index[0:Q.shape[0],:,:,0:Q.shape[3]], Q)
            
            R = jnp.zeros((r.shape[1] + 1, r.shape[1] + 1))
            R = jax.ops.index_update(R, jax.ops.index[:r.shape[0], :r.shape[1]], 0)
            for a in range(r.shape[0]):
                for b in range(r.shape[1]):
                    R = jax.ops.index_update(R, jax.ops.index[a,b], r[a,b])

            R = jax.ops.index_update(R, jax.ops.index[R.shape[0] - 1, R.shape[1] - 1], 1)


            W_prime[i+1] = jnp.tensordot(R, W_prime[i+1], 1)


        return W_prime

    @staticmethod
    def makeRightCanonical(self):

        W = JMPO.makeLeftCanonical(self)
        W_prime = JMPO.makeLeftCanonical(self)
    
        
        for i in range(len(W)):
            temp = W[len(W) - 1 - i].T
            
            for a in range(temp.shape[0]):
                for b in range(temp.shape[2]):
                    for c in range(temp.shape[1]):
                        W_prime[i] = jax.ops.index_update(W_prime[i], jax.ops.index[a,b,c], jnp.flip(temp[temp.shape[0] - 1 - a, c, b]))
        

        return W_prime

    @staticmethod
    def compress(self, bondDim):

        def dot(A, B):
            if len(A.shape) > len(B.shape):
                if A.shape[-1] != B.shape[0]:
                    raise Exception()

                result = jnp.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[1]))

                for a in range(A.shape[0]):
                    for b in range(A.shape[1]):
                        for c in range(A.shape[2]):
                            for d in range(B.shape[1]):
                                temp = 0
                                for e in range(A.shape[3]):
                                    temp += A[a,b,c,e] * B[e,d]
                                result = jax.ops.index_update(result, jax.ops.index[a,b,c,d], temp)



            else:
                if A.shape[-1] != B.shape[0]:
                    raise Exception()
                result = jnp.zeros((A.shape[0], B.shape[1], B.shape[2], B.shape[3]))

                for a in range(A.shape[0]):
                    for b in range(B.shape[1]):
                        for c in range(B.shape[2]):
                            for d in range(B.shape[3]):
                                temp = 0
                                for e in range(A.shape[1]):
                                    temp += A[a,e] * B[e,b,c,d]
                                result = jax.ops.index_update(result, jax.ops.index[a,b,c,d], temp)

            return result

        W = JMPO.makeRightCanonical(self)

        for i in range(len(self) - 1):
            #Get the upper left block of W[i]
            if i == 0:
                V = W[i][:,:,:,0:W[i].shape[3]-1]
            elif i == len(self) - 1:
                V = W[i][0:W[i].shape[0]-1,:,:,:]
            else:
                V = W[i][0:W[i].shape[0]-1,:,:,0:W[i].shape[3]-1]

            # Flatten V
            V_flatten = jnp.zeros((V.shape[0] * V.shape[1] * V.shape[2], V.shape[3]))
            counter = 0
            for a in range(V.shape[0]):
                for b in range(V.shape[1]):
                    for c in range(V.shape[2]):
                        for d in range(V.shape[3]):
                            V_flatten = jax.ops.index_update(V_flatten, jax.ops.index[counter, d], V[a,b,c,d])
                        counter += 1

            # Perform a QR Decomp on Flattened V
            q,r = jnp.linalg.qr(V_flatten)
            factor = q[0,0]
            q = q / factor
            r = r * factor

            if i == 0:
                Q = jnp.zeros((1,2,2,q.shape[-1]))
            else:
                Q = jnp.zeros(V.shape)

            counter = 0

            # Reshape Q
            for a in range(Q.shape[0]):
                for b in range(Q.shape[1]):
                    for c in range(Q.shape[2]):
                        Q = jax.ops.index_update(Q, jax.ops.index[a,b,c], q[counter])
                        counter += 1

            # Repopulate upper left block of W[i] with Q
            if Q.shape[-1] != W[i].shape[-1] - 1:
                temp = jnp.zeros(W[i].shape)
                temp = jax.ops.index_update(temp, jax.ops.index[:,:,:,-1], W[i][:,:,:,-1])
                temp = jax.ops.index_update(temp, jax.ops.index[:,:,:,:Q.shape[-1]], Q)
                W[i] = jnp.array(temp, copy = True)
            else:
                W[i] = jax.ops.index_update(W[i], jax.ops.index[0:Q.shape[0],:,:,0:Q.shape[3]], Q)


            # Expand R      
            R = jnp.zeros((r.shape[1] + 1, r.shape[1] + 1))
            for a in range(r.shape[0]):
                for b in range(r.shape[1]):
                    R = jax.ops.index_update(R, jax.ops.index[a,b], r[a,b])

            R = jax.ops.index_update(R, jax.ops.index[R.shape[0] - 1, R.shape[1] - 1], 1)

            # Split R into Msmall and R'
            Msmall = jax.ops.index_update(R, jax.ops.index[0,1:], 0)[1:R.shape[0] - 1,1:R.shape[1] - 1]
            R_prime = jax.ops.index_update(jnp.eye(R.shape[0], R.shape[1]), jax.ops.index[0,1:], R[0,1:])
            # SVD on Msmall
            u,s,v = jnp.linalg.svd(Msmall)


            # Cutoff small singular values and/or restrict bond dim
            #discard = min(len(s) - 1, maxDim - 2)
            discard = bondDim - 2

            # Expand U matrix
            U_expanded = jnp.zeros((u.shape[0] + 2, u.shape[1] + 2 - discard))
            U_expanded = jax.ops.index_update(U_expanded, jax.ops.index[0,0], 1)
            U_expanded = jax.ops.index_update(U_expanded, jax.ops.index[-1,-1], 1)

            U_expanded = jax.ops.index_update(U_expanded, jax.ops.index[1:1+u.shape[0], 1:1+u.shape[1] - discard], u[:,:u.shape[1] - discard])


            # Compress right link dimension of W[i]
            W[i] = dot(W[i], U_expanded)

            # Expand VT
            VT_expanded = jnp.zeros((v.shape[0] + 2 - discard, v.shape[1] + 2))
            VT_expanded = jax.ops.index_update(VT_expanded, jax.ops.index[0,0], 1)
            VT_expanded = jax.ops.index_update(VT_expanded, jax.ops.index[-1,-1], 1)

            for a in range(v.shape[1]):
                for b in range(v.shape[0] - discard):
                    VT_expanded = jax.ops.index_update(VT_expanded, jax.ops.index[b+1,a+1], v[b,a] * s[b])
            
            # Compress left link dimension of W[i+1]
            W[i+1] = dot(VT_expanded @ R_prime, W[i+1])



        return W

    # This method will add two JMPOs and spit back out the sum compressed to the dimensions of the first JMPO
    @staticmethod
    def add(jmpo1, jmpo2):
        W = []
        A = jnp.array(jnp.moveaxis(jmpo1[0], 3, 1), copy = True)
        B = jnp.array(jnp.moveaxis(jmpo2[0], 3, 1), copy = True)
        C = jnp.zeros((1, A.shape[1] - 1 + B.shape[1] - 1, A.shape[2], A.shape[3]))
        C = jax.ops.index_update(C, jax.ops.index[:,:A.shape[1] - 1], A[0,:A.shape[1] - 1])
        C = jax.ops.index_update(C, jax.ops.index[:,A.shape[1] - 1:C.shape[1] - 1], B[0,1:B.shape[1] - 1])
        C = jax.ops.index_update(C, jax.ops.index[0,-1], A[0,-1] + B[0,-1])
        C = jnp.moveaxis(C,1,3)
        W.append(C)


        for i in range(1,len(jmpo1)-1):
            A = jnp.array(jnp.moveaxis(jmpo1[i], 3, 1), copy = True)
            B = jnp.array(jnp.moveaxis(jmpo2[i], 3, 1), copy = True)
            C = jnp.zeros((A.shape[0] - 1 + B.shape[0] - 1, A.shape[1] - 1 + B.shape[1] - 1, A.shape[2], A.shape[3]))
            C = jax.ops.index_update(C, jax.ops.index[:A.shape[0] - 1, :A.shape[1] - 1], A[:A.shape[0] - 1, :A.shape[1] - 1])
            C = jax.ops.index_update(C, jax.ops.index[A.shape[0] - 1:,A.shape[1] - 1:], B[1:,1:])
            C = jax.ops.index_update(C, jax.ops.index[0, A.shape[1] - 1:C.shape[1] - 1], B[0,1:B.shape[1] - 1])
            C = jax.ops.index_update(C, jax.ops.index[1:A.shape[0] - 1,-1], A[1:A.shape[0] - 1,-1])
            C = jax.ops.index_update(C, jax.ops.index[0,-1], A[0,-1] + B[0,-1])
            C = jnp.moveaxis(C,1,3)

            W.append(C)
        
        A = jnp.array(jnp.moveaxis(jmpo1[-1], 3,1), copy = True)
        B = jnp.array(jnp.moveaxis(jmpo2[-1], 3,1), copy = True)
        C = jnp.zeros((A.shape[0] - 1 + B.shape[0] - 1, 1, A.shape[2], A.shape[3]))
        C = jax.ops.index_update(C, jax.ops.index[0,0], A[0,-1] + B[0,-1])
        C = jax.ops.index_update(C, jax.ops.index[1:A.shape[0] - 1,0], A[1:A.shape[0] - 1,-1])
        C = jax.ops.index_update(C, jax.ops.index[A.shape[0] - 1:,0], B[1:,-1])
        C = jnp.moveaxis(C,1,3)
        W.append(C)

        bondDim = jmpo1[0].shape[-1]

        return JMPO.compress(W, bondDim)