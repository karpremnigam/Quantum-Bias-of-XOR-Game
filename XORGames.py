import numpy as np 
import cvxpy

class XORGame:
    def __init__(self, pi: np.matrix, f: np.matrix) -> None:
        self.pi = pi
        self.f = f
        x, y = self.pi.shape

        if np.min(self.pi) < 0:
            raise ValueError("Not a valid probability distribution: Contains negative values.") 
        
        if np.sum(self.pi) != 1:
            raise ValueError("Not a valid probability distribution: Entries don't sum to one.")
        
        if (x,y) != self.f.shape:
            raise ValueError("Inconsistent probability distribution and function.")
        
        if np.sum(self.f[self.f<0]) != 0:
            raise ValueError("The range of the function should ony contain 0, 1")
        
        if np.sum(self.f[self.f > 1]) != 0:
            raise ValueError("The range of the function should ony contain 0, 1")
        
        if np.sum(self.f[self.f > 0]) != np.sum(self.f[self.f >=1]):
            raise ValueError("The range of the function should ony contain 0, 1")
        
    def QuantumValue(self) -> float:
        x,y = self.pi.shape
        d = np.zeros([x,y])

        for i in range(x):
            for j in range(y):
                d[i,j] = self.pi[i,j]*(-1)**(self.f[i,j])

        u = cvxpy.Variable(x, complex = False)
        v = cvxpy.Variable(y, complex = False)

        objective = cvxpy.Minimize(cvxpy.sum(u) + cvxpy.sum(v))
        constraints = [cvxpy.bmat([[cvxpy.diag(u), -d], [-np.transpose(np.conjugate(d)), cvxpy.diag(v)]])>>0]

        problem = cvxpy.Problem(objective, constraints)
        problem.solve()

        return np.real(problem.value)/4+1/2

        

