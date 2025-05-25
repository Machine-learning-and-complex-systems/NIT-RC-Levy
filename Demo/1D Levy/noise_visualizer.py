# noise_visualizer.py
"""
Input requirements:
    - original_noise: Original noise, (n_samples,) NumPy array
    - separated_noise: Separated noise, (n_samples,) NumPy array
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  levy_stable, norm

plt.style.use('seaborn-v0_8-deep')

def plot_levy_comparison(original_noise, separated_noise, 
                        save_path=None, figsize=(14, 6), 
                        dpi=150, bins=200, xlim=(-5,5)):

    
    # 输入校验
    original = np.asarray(original_noise).flatten()
    separated = np.asarray(separated_noise).flatten()
    print("np.max(original):", np.max(original))
    print("np.max(separated):", np.max(separated))
    
    if len(original) == 0 or len(separated) == 0:
        raise ValueError("Input data cannot be empty")
    
   
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.suptitle("Lévy Noise Distribution Analysis", y=1.02, fontsize=14, weight='bold')
    
    # ----------------- QQ -----------------
    ax1 = plt.subplot2grid((1,2), (0,0))
    
    # Calculate quantiles
    q = np.linspace(0, 1, min(len(original), len(separated)))
    orig_quantiles = np.quantile(original, q)
    sep_quantiles = np.quantile(separated, q)
    
    #  Plot QQ line
    ax1.scatter(orig_quantiles, sep_quantiles, s=10, c='#018571', alpha=0.6, edgecolor='none')
    ax1.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], '--', lw=1, color='#444444')
    
    ax1.set_xlabel('Original Quantiles', fontsize=9)
    ax1.set_ylabel('Separated Quantiles', fontsize=9)
    ax1.set_title('Q-Q Plot', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ----------------------------------
    ax2 = plt.subplot2grid((1,2), (0,1))

    
    
    # logscale
    
    hist_args = dict(bins=bins, density=True, histtype='step', linewidth=1.5)
    ax2.hist(original, **hist_args, color='#2c7bb6', label='Original')
    ax2.hist(separated, **hist_args, color='#d7191c', linestyle='--', label='Separated')
    
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Noise Amplitude', fontsize=9)
    ax2.set_ylabel('Log Density', fontsize=9)
    ax2.set_title('Tail Behavior (Log Scale)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20,20)
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

if __name__ == "__main__":
    # Example usage ----------
    # Generate test data (Lévy stable distribution)
    alpha = 1.5
    beta = 0
    n_samples = 10000
    np.random.seed(42)
    
    original = levy_stable.rvs(alpha, beta, size=n_samples)
    separated = original + 0.1*np.random.randn(n_samples)  # Add small amount of noise to simulate separated noise
    print("original.shape:", original.shape)
    print("separated.shape:", separated.shape)
    # Call plotting function
    plot_levy_comparison(original, separated,
                        save_path="levy_comparison.png",
                        xlim=(-10,10))
