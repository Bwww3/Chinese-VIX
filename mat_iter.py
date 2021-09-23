# -*- coding: utf-8 -*-

import numpy as np


#%%
class LinearEqIterator():
    
    def __init__(self, eps=1e-4, max_iter=100):
        self.eps = eps
        self.max_iter = max_iter


#%%
class FixedPointIterator(LinearEqIterator):
    
    def __init__(self, method="Jacobi", eps=1e-4, max_iter=100, **args):
        super().__init__(eps=eps, max_iter=max_iter)
        self.method = method
        self.args = args
        
    def solve(self, A, b, x0):
        D = np.diag(np.diag(A))
        L, U = -np.tril(A, -1), -np.triu(A, 1)
        mapping = {
            "Jacobi": self.Jacobi_method,
            "GS":     self.Gaussian_Seidel_method,
            "SOR":    self.SOR_method
            }
        R, c = mapping[self.method](D, L, U, **self.args)
        rhoR = np.abs(np.max(np.linalg.eigvals(R)))
        n = 1; xn = x0; xs = []
        while n <= self.max_iter:
            xnp1 = np.dot(R, xn) + c
            xs.append(xnp1)
            if np.linalg.norm(xnp1-xn) < self.eps:
                break
            else:
                xn = xnp1
                n += 1
        return xs[-1], R, c, xs, rhoR
            
    def Jacobi_method(self, D, L, U):
        R = np.dot(np.linalg.inv(D), L + U)
        c = np.dot(np.linalg.inv(D), b)
        return R, c
    
    def Gaussian_Seidel_method(self, D, L, U):
        R = np.dot(np.linalg.inv(D-L), U)
        c = np.dot(np.linalg.inv(D-L), b)
        return R, c
    
    def SOR_method(self, D, L, U, w=1.0):
        R = np.dot(np.linalg.inv(D-w*L), (1-w)*D + w*U)
        c = w * np.dot(np.linalg.inv(D-w*L), b)
        return R, c
        
    
#%%
class GradientIterator(LinearEqIterator):
    
    def __init__(self, method="SD", eps=1e-4, max_iter=100, **args):
        super().__init__(eps=eps, max_iter=max_iter)
        self.method = method
        self.args = args
        
    def solve(self, A, b, x0):
        try:
            self.eligibility_check(A)
        except:
            print("Matrix A should be symmetric positive definit!")
            return
        mapping = {
            "SD": self.steepest_descent_method,
            "CG": self.conjugate_gradient_method
            }
        return mapping[self.method](A, b, x0)
    
    def steepest_descent_method(self, A, b, x0):
        n = 1; xn = x0; xs = []
        while n <= self.max_iter:
            rn = np.dot(A, xn) - b
            if np.linalg.norm(rn) > self.eps:
                an = np.vdot(rn, rn)[0, 0] / np.vdot(np.dot(A, rn), rn)[0, 0]
                xnp1 = xn - np.dot(an, rn)
                xs.append(xnp1)
                xn = xnp1
                n += 1
            else:  
                break
        return xs[-1], xs
    
    def conjugate_gradient_method(self, A, b, x0):
        pass
    
    @staticmethod
    def eligibility_check(A):
        assert((np.linalg.eigvals(A) > 0).all())
        assert((np.tril(A, -1).T == np.triu(A, 1)).all())
 
            
# In[]
if __name__ == "__main__":
    A = np.mat([[5, 1, 1],
                [1, 5, 1],
                [1, 1, 5]])
    b = np.mat([7 ,7, 7]).T
    x0 = np.mat([0.0, 0.0, 0.0]).T
    
    iterator = FixedPointIterator(method="SOR", max_iter=3, w=1.2)
    x, R, c, xs, rhoR = iterator.solve(A, b, x0)

    gradient_iterator = GradientIterator(method="SD")
    x, xs = gradient_iterator.solve(A, b, x0)
    










    
        
        
    
    
