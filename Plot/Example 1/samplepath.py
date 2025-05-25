import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FuncFormatter
import os
from sde_simulator import SDESimulator
from scipy.stats import levy_stable, norm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 2
current_dir = os.path.dirname(__file__)

colors = ['#F5A889', '#ACD6EC']

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  

# First one subplot
sim = SDESimulator()
para1 = [1.5, 0.5, 1, 100, 1e-3]  # [alpha, X0, eps, T, dt]
u1, dL1 = sim.simulate_sde(para1, 'levy')
dt1 = para1[4]  # dt
x1 = np.arange(len(u1)) * dt1  # create x axis array and multiple by dt
axes[0].plot(x1, u1, label='Lévy', color=colors[0])  
axes[0].set_xlim(0, 10000 * dt1)  
axes[0].set_ylim(-2, 2)
axes[0].set_xlabel('Time') 
axes[0].set_ylabel(r'$u_{t}$')
axes[0].legend(loc='upper right')  
axes[0].text(0.01, 0.99, '(a)', transform=axes[0].transAxes, fontsize=15, va='top', ha='left')  

# The second one subplot
sim = SDESimulator()
para2 = [2, 0.5, 1, 100, 1e-3]  # [alpha, X0, eps, T, dt]
u2, dL2 = sim.simulate_sde(para2, 'gaussian')
dt2 = para2[4]  
x2 = np.arange(len(u2)) * dt2  
axes[1].plot(x2, u2, label='Gaussian', color=colors[1])  
axes[1].set_xlim(0, 10000 * dt2) 
axes[1].set_ylim(-2, 2)
axes[1].set_xlabel('Time')  
axes[1].set_ylabel(r'$u_{t}$')
axes[1].legend(loc='upper right')  
axes[1].text(0.01, 0.99, '(b)', transform=axes[1].transAxes, fontsize=15, va='top', ha='left')  

# The third one subplot
# calculate the histogram of simulated data
dt = 1e-3
eps = 1
alpha = 1.5
levy_scale = eps * (dt)**(1/alpha)  # ≈0.01
gauss_std = eps * np.sqrt(dt)       # ≈0.0316

# 生成理论曲线数据
boundary = 2
x_levy = np.linspace(-boundary, boundary, 10000)  # Lévy's scale
x_gauss = np.linspace(-boundary, boundary, 10000)
levy_pdf = levy_stable.pdf(x_levy, alpha=alpha, beta=0, scale=levy_scale)
gauss_pdf = norm.pdf(x_gauss, scale=gauss_std)

# 绘制直方图（调整分箱）
axes[2].hist(dL1, bins=1000, range=(-boundary, boundary), density=True,
             color='#F5A889', alpha=0.4, label='Lévy simulated')
axes[2].hist(dL2, bins=1000, range=(-boundary, boundary), density=True,
             color='#ACD6EC', alpha=0.4, label='Gaussian simulated')

# 绘制理论曲线
axes[2].plot(x_levy, levy_pdf, color='#8B0000', lw=2, label='Theoretical Lévy')
axes[2].plot(x_gauss, gauss_pdf, color='#00008B', lw=2, linestyle='--', label='Theoretical Gaussian')
axes[2].set_xlim(-0.75, 0.75)  
axes[2].set_ylim(1e-3, 100)  
axes[2].set_yscale('log') 
axes[2].set_xlabel('Noise Amplitude', fontsize=15)
axes[2].set_ylabel('PDF', fontsize=15)
axes[2].legend(fontsize=15, loc='upper right')  
axes[2].text(0.01, 0.99, '(c)', transform=axes[2].transAxes, fontsize=15, va='top', ha='left')  


plt.tight_layout()

plt.savefig('combined_figure.pdf', bbox_inches='tight')
plt.show()