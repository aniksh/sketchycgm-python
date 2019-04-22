
# coding: utf-8

# In[63]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import LinearOperator, svds
import scipy
import time, datetime
import pickle


# In[64]:


def nuclear_norm(A):
    """Nuclear norm of input matrix"""
    return np.sum(np.linalg.svd(A)[1])

def mat2im(X):
    """
    Input:
        X - A numpy 2d array
        
    Output:
        A Pillow image in the range 0-255
    """
    Xi = X.copy()
    Xmin = np.min(Xi)
    Xi -= Xmin
    Xi = Xi / np.max(Xi)
    return Image.fromarray(np.uint8(Xi*255))


# In[2]:


# m = 1000
# n = 1200
# r = 20
# X = np.zeros((m,n))
# for i in range(r):
#     a = np.random.randn(m,1)
#     b = np.random.randn(1,n)
#     X += a.dot(b)


# In[5]:


# d = int(0.1*m*n)
# samples = sorted(np.random.permutation(np.arange(m*n))[:2*d])
# E = np.zeros((d, 2), dtype=np.int32)
# for idx, i in enumerate(samples[:d]):
#     E[idx,0] = i//n
#     E[idx,1] = i%m

# Etest = np.zeros((d, 2), dtype=np.int32)
# for idx, i in enumerate(samples[d:2*d]):
#     Etest[idx,0] = i//n
#     Etest[idx,1] = i%m


# In[6]:


# b = X[E[:,0], E[:,1]].copy()
# btest = X[Etest[:,0], Etest[:,1]].copy()


# In[57]:


# data = np.load('../hw2/hw2recoverydata.npz')
# m = data['m']
# n = data['n']
# tau = data['tau']
# rank = data['rank']
# E = np.unique(data['Omega'], axis=0)
# b = data['y']
# print(m,n)
# print(tau, rank)
# print(E.shape, b.shape)


# # In[88]:


# plt.imshow(mapper.vec2mat(b), cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.savefig('sampled_image', dpi=200)

m = 943
n = 1682

def load_ml(split):
    E = []
    b = []
    with open('ml-100k/ub.' + split) as f:
        for line in f:
            i,j,v, _ = line.split()
            E.append((i,j))
            b.append(v)

    E = np.array(E, dtype=np.int32) - 1
    b = np.array(b, dtype=np.int32)
    return E,b

E, b = load_ml('base')
Etest, btest = load_ml('test')

print(E.shape)
assert np.min(E) == 0 and np.max(E) == n-1


def binarize(x, t):
    return (x>t) * 1 + (x<t) * (-1)

b = binarize(b, 3.5)
btest = binarize(btest, 3.5)

# In[39]:


class AdjointOperator(LinearOperator):
    
    def __init__(self, shape, Sigma, x, dtype=np.float32):
        """
        Creates a LinearOperator object from observations
        
        Args:
            shape: tuple
            Sigma: numpy.ndarray (len(x),2), The indices of the 
                samples from original matrix
            x: numpy.ndarray, The observations
        
        Returns:
            The adjoint map of the vector x to a matrix corresponding
            to Sigma
        """
        assert len(x) == Sigma.shape[0]
        self.shape = shape
        self.E = Sigma
        self.x = x
        self.dtype = dtype
        
    def _matvec(self, v):
        assert v.shape[0] == self.shape[1]
        res = np.zeros((self.shape[0], )) if len(v.shape)==1 else np.zeros((self.shape[0], 1))
        for i, idx in enumerate(self.E):
            res[idx[0]] += self.x[i] * v[idx[1]]
        
        return res
    
    def _matmat(self, V):
        assert V.shape[0] == self.shape[1]
        res = np.zeros(self.shape[0], V.shape[1])
        for i, idx in enumerate(self.E):
            res[idx[0], :] += self.x[i] * V[idx[1], :]
        
        return res
    
    def _rmatvec(self, v):
        assert v.shape[0] == self.shape[0]
        res = np.zeros((self.shape[1], )) if len(v.shape)==1 else np.zeros((self.shape[1], 1))
        for i, idx in enumerate(self.E):
            res[idx[1]] += self.x[i] * v[idx[0]]
        
        return res


# In[9]:


def test_adjoint_map():
    sigma = np.array([(0,1), (0,2), (1,1), (2,0)])
    z = np.arange(4)
    print(sigma, z)
    A = np.zeros((3,3))
    for i, idx in enumerate(sigma):
        A[idx[0], idx[1]] = z[i]
    
    adjz = AdjointOperator((3,3), sigma, z)
    v = np.arange(5,8)
    print('A', A)
    print('v',v)
    print('A*v', A@v)
    print('adj(z)*v', adjz@v)
    print('A.H*v', A.T@v)
    print('adj(z).H*v', adjz.H@v)
    
# test_adjoint_map()


# In[10]:


class LinearMap():
    
    def __init__(self, Sigma):
        self.E = Sigma
        
    def __call__(self, u, v):
        """
        Args:
            u: vector
            v: vector
            
        Returns: 
            map(uv.T)
        """
        res = np.zeros((self.E.shape[0],))
        for i, idx in enumerate(self.E):
            res[i] = u[idx[0]] * v[idx[1]]
        
        return res


# In[66]:


class Mapping():
    def __init__(self, shape, Sigma):
        shape = shape
        self.E = Sigma
        
    def vec2mat(self, vec):
        res = np.zeros((m,n))
        for i, idx in enumerate(self.E):
            res[idx[0], idx[1]] = vec[i]

#         return scipy.sparse.csr_matrix(res)
        return res

    def mat2vec(self, mat):
        res = np.zeros(len(self.E))
        for i, idx in enumerate(self.E):
            res[i] = mat[idx[0], idx[1]]

        return res

mapper = Mapping((m,n), E)
testmapper = Mapping((m,n), Etest)

# In[12]:


def loss_gauss(z, b):
    return 1/ len(b) * 0.5 * np.sum((z-b)**2)

def gradient_gauss(z, b):
    return 1/len(b) * (z-b)

def test_loss(X, loss):
    return loss(testmapper.mat2vec(X), btest)

# In[13]:


def CGM(shape, b, test_loss, loss, gradient, mapper, alpha, tol, n_iter):
    Xt = np.zeros(shape)
    losses = []
    test_losses = []
    svals = {}
        
    for t in range(n_iter):
        st = time.time()
        zt = mapper.mat2vec(Xt)
        
        df = gradient(zt, b)
        u, _, v = svds(mapper.vec2mat(df), 1)
        Ht = - alpha * u.dot(v)
    
        if np.dot(zt - mapper.mat2vec(Ht), gradient(zt, b)) <= tol:
            break
    
        eta = 2 / (t+2)
        Xt = (1 - eta) * Xt + eta * Ht        
        
        zt = mapper.mat2vec(Xt)
        l = loss(zt,b)
        losses.append(l)
        test_losses.append(test_loss(Xt, loss))
        et = time.time()
        print("iteration: {:4d}\t time/iteration: {:2.4f}\t loss: {:0.4f} ".format(t+1, et -st, l), end='\r')
        if (t + 1) % 10 == 0:
            _, s, _ = svds(Xt,300)
            svals[t+1] = s
        
    print()
    return Xt, losses, test_losses, svals


# In[73]:

alpha = 4000
print('alpha:', alpha)
st = time.time()
Xcgm, losses_cgm, test_losses_cgm, svals = CGM((m,n), b, test_loss, loss_gauss, gradient_gauss, mapper, alpha=alpha, tol=1e-3, n_iter=500)

# print('||X-Xcgm||_F', np.linalg.norm(X-Xcgm))
print('total time', datetime.timedelta(seconds=time.time() - st))

print('test loss', loss_gauss(Xcgm[Etest[:,0], Etest[:,1]], btest))

# # plt.figure(1)
# # plt.semilogx(range(50,5000), losses_cgm[50:], label='CGM')
# # plt.figure(2)
# # plt.semilogx(range(50,5000), test_losses_cgm[50:], label='CGM')
# # # plt.show()
with open('loss-cgm.pkl', 'wb') as f:
    pickle.dump((Xcgm, losses_cgm, test_losses_cgm, svals), f)

# In[87]:


# plt.figure(figsize=(12,8))
# plt.imshow(mat2im(Xcgm))
# plt.xticks([]), plt.yticks([])
# plt.savefig('cgm_image', dpi=200)
# plt.show()


# In[15]:



# In[24]:


class CGMSketch():
    """Sketch of a low-rank matrix"""
    
    def __init__(self, shape, rank, dtype=np.float32):
        """@todo: to be defined1.
        """
        self.shape = shape
        self.dtype = dtype
        self.rank = rank

        k = 2 * rank + 1
        l = 2 * k + 1
        self.Omega = np.random.randn(shape[1], k)
        self.Psi = np.random.randn(l, shape[0])

        self.Y = np.zeros((shape[0], self.Omega.shape[1]), dtype=dtype)
        self.W = np.zeros((self.Psi.shape[0], shape[1]), dtype=dtype)

    def cgm_update(self, eta, u, v):
        """Performs an in-place update of the sketched matrix `X` equivalent to
            X <- (1 - eta) X + eta * alpha (u * v)
        `u` and `v` are expected in the same format as the come out of
        :func:`linalg.svds`.
        :param float eta: Update weight
        :param u: Vector of length `X.shape[0]`
        :param v: Vector of length `X.shape[1]`
        :param float alpha:
        :returns: `self`
        """
        self.Y = (1 - eta) * self.Y + eta * u.dot(v.dot(self.Omega))
        self.W = (1 - eta) * self.W + eta * (self.Psi.dot(u)).dot(v)

    def factorization(self):
        Q, _ = np.linalg.qr(self.Y)
        B = np.linalg.pinv(self.Psi.dot(Q)).dot(self.W)
        return Q, B

    def reconstruct(self):
        """@todo: Docstring for recons.
        """
        Q, B = self.factorization()
        U, S, VT = svds(B, self.rank)
        return Q.dot(U), S, VT


# In[79]:


def sketchyCGM(shape, b, test_loss, loss, gradient, linmap, adjmap, rank, alpha, tol, n_iter):
    """
    Performs SketchyCGM
    
    Args:
        shape: shape of the solution matrix
        b: Target values
        loss: loss function
        gradient: gradient function
        linmap: Map from X to z
        adjmap: Adjoint map operator from z to X
        rank: rank of solution
        alpha: nuclear norm of solution
        tol: suboptimality epsilon
        
    Returns:
        Solution matrix: U, Sigma, VT
    """
    # Sketch.INIT
    sketch = CGMSketch(shape, rank)
    z = np.zeros_like(b)
    losses = []
    test_losses = []

    for t in range(n_iter):
        st = time.time()
        dz = gradient(z, b)
        u, _, v = svds(adjmap(dz), 1)
        h = linmap(-alpha*u, v.T)
        
        if np.dot(z-h, dz) <= tol:
            break
            
        eta = 2/(t+2)
        z = (1-eta)*z + eta*h        
        sketch.cgm_update(eta, -alpha * u, v)
        
        l = loss(z,b)
        losses.append(l)
        et = time.time()
        print("iteration: {:5d}\t time/iteration: {:2.4f}\t loss: {:0.4f}".format(t+1, et -st, l), end='\r')
        U,S,VT = sketch.reconstruct()
        test_losses.append(test_loss(U.dot(np.diag(S).dot(VT)), loss))
        
    print()
    return sketch.reconstruct(), losses, test_losses


# In[42]:


class SigmaOp(AdjointOperator):
    """
    Initialize adjoint map with E and (m,n)
    """
    def __init__(self, x, dtype=np.float32):
        super(SigmaOp,self).__init__(shape=(m,n), Sigma=E, x=x, dtype=dtype)


# In[80]:


linmap = LinearMap(E)
st = time.time()
(U,S,VT), losses, test_losses = sketchyCGM((m,n),
                                          b,
                                          test_loss,
                                          loss_gauss, 
                                          gradient_gauss, 
                                          linmap, 
                                          mapper.vec2mat, 
                                          rank=50, 
                                          alpha=alpha, 
                                          tol=1e-3,
                                          n_iter=5000)

Xsol = U.dot(np.diag(S).dot(VT))

# print('||X-Xsol||_F', np.linalg.norm(X-Xsol))
print('total time', datetime.timedelta(seconds=time.time() - st))

# print('test loss', loss_gauss(Xsol[Etest[:,0], Etest[:,1]], btest))
# plt.figure(1)
# plt.semilogx(range(50,5000), losses[50:], label='SketchyCGM')
# plt.title('Quadratic Loss'), plt.xlabel('iterations'), plt.ylabel('loss')
# plt.savefig('loss.png', dpi=200)
# plt.figure(2)
# plt.semilogx(range(50,5000), test_losses[50:], label='SketchyCGM')
# plt.title('Quadratic Loss'), plt.xlabel('iterations'), plt.ylabel('test-error')
# plt.savefig('test-error.png', dpi=200)
# plt.show()


with open('loss-r50.pkl', 'wb') as f:
    pickle.dump((Xsol, losses, test_losses), f)

# # In[86]:


# plt.figure(figsize=(12,8))
# plt.imshow(mat2im(Xsol))
# plt.xticks([]), plt.yticks([])
# plt.savefig('scgm_image', dpi=200)
# plt.show()

