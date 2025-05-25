# 新建文件：sde_simulator.py
import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
class SDESimulator:
    # def __init__(self, seed=42):
    #     np.random.seed(seed)
        
    def generate_levy_increments(self, alpha, dt, n):
      beta = 0  # 对称稳定分布
      scale = (dt)**(1 / alpha)
      increments = levy_stable.rvs(alpha, beta, scale=scale, size=n, random_state=42)
    #   increments = levy_stable.rvs(alpha, beta, scale=scale, size=n)
    
      # 动态截断阈值（保留99.9%数据特征）
      abs_inc = np.abs(increments)
      q = np.quantile(abs_inc, 0.999)
      return np.clip(increments, -20*q, 20*q)

    def drift(self, x):
        """双势阱漂移项"""
        return -(-x + x**3)

    def simulate_sde(self, para, noise_type='gaussian'):
        """
        通用SDE模拟函数
        参数：
            para : 包含 [alpha, X0, eps, T, dt] 的列表
            noise_type : 'gaussian' 或 'levy'
        """
        alpha, X0, eps, T, dt = para
        n = int(T / dt)
        X = np.zeros(n + 1)
        X[0] = X0
        
        if noise_type == 'levy':
            dL = self.generate_levy_increments(alpha, dt, n)
        else:  # 高斯噪声
            dL = np.sqrt(dt) * np.random.randn(n)
            
        for i in range(n):
            # 四阶Runge-Kutta方法
            k1 = self.drift(X[i]) * dt
            k2 = self.drift(X[i] + 0.5 * k1) * dt
            k3 = self.drift(X[i] + 0.5 * k2) * dt
            k4 = self.drift(X[i] + k3) * dt
            X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)/6 + eps*dL[i]
            
        return X, dL

# 兼容原有DoubleWell接口
def DoubleWell(para, noise_type='gaussian'):
    """
    兼容原有调用的接口
    参数：
        para : [alpha, X0, eps, T, dt]
               对于高斯噪声，alpha参数会被忽略
    """
    sim = SDESimulator()
    return sim.simulate_sde(para, noise_type)[0]  # 只返回X序列



if __name__ == "__main__":
    # 示例用法
    sim = SDESimulator()
    para = [1.5, 0.5, 1, 50, 1e-3]
    X, dL = sim.simulate_sde(para, noise_type='levy')
    plt.plot(X)
    plt.xlim(0,10000)
    plt.ylim(-2,2)
    plt.show()