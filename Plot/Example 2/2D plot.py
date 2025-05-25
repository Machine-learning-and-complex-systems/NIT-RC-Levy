import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from LCload import load_2d_dataset


def create_composite_figure(dataset, save_name="composite_plot.png"):
    # Set figure parameters
    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'font.family': 'Times New Roman',
        'axes.linewidth': 1.5,
    })

    # Create layout using GridSpec to achieve the desired layout
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[3, 2, 1.5])

    dt = 0.01

    # Panel 1: Time series plot
    ax1 = fig.add_subplot(gs[0, :])
    train = dataset.train.T
    test = dataset.test.T
    pred = dataset.pred.T

    
    train_time = np.linspace(0, 5000 * dt, len(train))
    test_time = np.linspace(5000 * dt, 11000 * dt, len(test))
    pred_time = np.linspace(5000 * dt, 5000 * dt + 5000 * dt, 5000)

   

    ax1.plot(train_time, train[:, 0], color='salmon', label='Training X')
    ax1.plot(pred_time, pred[:5000, 0], color='lightblue', label='Predicted X')
    ax1.plot(train_time, train[:, 1], '--', color='salmon', label='Training Y', alpha=0.6)
    ax1.plot(pred_time, pred[:5000, 1], '--', color='lightblue', label='Predicted Y', alpha=0.6)
    ax1.axvline(5000 * dt, color='gray', ls='--', lw=1)
    ax1.set(xlabel='Time', ylabel='X and Y', xlim=(0, 100))
    ax1.set_ylim(-3,3)
    ax1.legend(ncol=4, loc='upper right')
    ax1.text(-0.04, 1.1, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')


        # Panel 2: Scatter plot 
    ax2 = fig.add_subplot(gs[1, 0])
    
    pred_same_length = pred[:5000]
    train_x = train[:5000, 0]
    train_y = train[:5000, 1]
    pred_x = pred_same_length[:, 0]
    pred_y = pred_same_length[:, 1]
    # Generate time values for color mapping
    time_values_train = np.linspace(0, 50, 5000)
    time_values_pred = np.linspace(50, 100, 5000)

    
    sc_train = ax2.scatter(train_x, train_y, c=time_values_train, cmap='viridis', label='Training Data', alpha=0.6, marker='o')
    
    sc_pred = ax2.scatter(pred_x, pred_y, c=time_values_pred, cmap='viridis', label='Predicted Data', marker='x', alpha=0.6)

    ax2.set(xlabel='X', ylabel='Y')
    ax2.legend()
    ax2.text(-0.1, 1.1, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')
  
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=100)), ax=ax2, label='Time')

   
    ax3 = fig.add_subplot(gs[1, 1])
    noise_y = dataset.noise_y

 
    n, bins, _ = ax3.hist(noise_y, bins=50, density=True, color='lightblue', alpha=0.6, label='Separated Noise (Y-direction)')

    # Theoretical noise distribution standard normal * sqrt(dt)
    gauss_std = np.sqrt(dt)
    x_gauss = np.linspace(-3 * gauss_std, 3 * gauss_std, 100)
    gauss_pdf = norm.pdf(x_gauss, scale=gauss_std)
    ax3.plot(x_gauss, gauss_pdf, color='salmon', linestyle='--', label='Theoretical Noise')

    ax3.set_ylim(1e-1, )
    ax3.set(xlabel='Noise Amplitude', ylabel='PDF')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.text(-0.1, 1.1, '(c)', transform=ax3.transAxes,  fontweight='bold', va='top')




    # 面板 4: Transition time distribution
    ax4 = fig.add_subplot(gs[1, 2])
    times_true = dataset.times_true * dt
    times_pred = dataset.times_pred * dt
    counts_true = int(dataset.counts_true)
    counts_pred = int(dataset.counts_pred)

  
    left = min(np.min(times_true), np.min(times_pred))
    right = max(np.max(times_true), np.max(times_pred))
    bins = np.arange(math.floor(left), math.ceil(right), 1)

    ax4.hist([times_true, times_pred], bins=bins,
             density=True, color=['salmon', 'lightblue'])
    ax4.set(xlabel=' Time', ylabel='PDF')

  
    legend_text = [
        f'Test: {counts_true} (μ={np.mean(times_true):.2f})',
        f'Pred: {counts_pred} (μ={np.mean(times_pred):.2f})'
    ]
    ax4.legend(legend_text, title='Transition Counts', loc='upper right')
    ax4.text(-0.1, 1.1, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')
    
    plt.tight_layout()

        # 修改保存路径和文件格式
    new_save_name = "limit cycle.pdf"
    save_path = os.path.join(os.path.dirname(__file__), new_save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    dataset = load_2d_dataset()
    create_composite_figure(dataset)
    