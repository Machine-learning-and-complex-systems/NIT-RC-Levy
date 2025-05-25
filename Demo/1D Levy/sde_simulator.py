# ：sde_simulator.py
import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
class SDESimulator:
    # def __init__(self, seed=42):
    #     np.random.seed(seed)
        
    def generate_levy_increments(self, alpha, dt, n):
      beta = 0  # symmetric stable
      scale = (dt)**(1 / alpha)
      increments = levy_stable.rvs(alpha, beta, scale=scale, size=n, random_state=42)
    #   increments = levy_stable.rvs(alpha, beta, scale=scale, size=n)
    
      # Dynamic truncation threshold (preserve 99.9% data characteristics)
      abs_inc = np.abs(increments)
      q = np.quantile(abs_inc, 0.999)
      return np.clip(increments, -20*q, 20*q)

    def drift(self, x):
        """double well drift function"""
        return -(-x + x**3)

    def simulate_sde(self, para, noise_type='gaussian'):
        """
        sde simulation
        parameters：
            para :  [alpha, X0, eps, T, dt] 
            noise_type : 'gaussian' or 'levy'
        """
        alpha, X0, eps, T, dt = para
        n = int(T / dt)
        X = np.zeros(n + 1)
        X[0] = X0
        
        if noise_type == 'levy':
            dL = self.generate_levy_increments(alpha, dt, n)
        else:  # Gaussian noise
            dL = np.sqrt(dt) * np.random.randn(n)
            
        for i in range(n):
            # Runge-Kutta
            k1 = self.drift(X[i]) * dt
            k2 = self.drift(X[i] + 0.5 * k1) * dt
            k3 = self.drift(X[i] + 0.5 * k2) * dt
            k4 = self.drift(X[i] + k3) * dt
            X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)/6 + eps*dL[i]
            
        return X, dL

def DoubleWell(para, noise_type='gaussian'):
    """
    parameters：
        para : [alpha, X0, eps, T, dt]
              
    """
    sim = SDESimulator()
    return sim.simulate_sde(para, noise_type)[0]  # 



if __name__ == "__main__":
    # Example usage
    sim = SDESimulator()
    para = [1.5, 0.5, 1, 50, 1e-3]
    X, dL = sim.simulate_sde(para, noise_type='levy')
    plt.plot(X)
    plt.xlim(0,10000)
    plt.ylim(-2,2)
    plt.show()