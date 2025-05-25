import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, norm


# 加载数据的函数
def safe_load(file_path):
    try:
        data = np.load(file_path)
        print(f"✔ Successfully loaded {os.path.basename(file_path)} with shape {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ File {os.path.basename(file_path)} not found")
        return None


def load_1d_data():
    data_folder = r'C:\Users\LENOVO\Desktop\RC for Levy\data\1D Levy'
    train_data = safe_load(os.path.join(data_folder, "original data.npy"))
    test_data = safe_load(os.path.join(data_folder, "test_data.npy"))
    pred_data = safe_load(os.path.join(data_folder, "predicted data.npy"))
    trans_counts_true = safe_load(os.path.join(data_folder, "transition_counts_true.npy"))
    trans_counts_pred = safe_load(os.path.join(data_folder, "transition_counts_pred.npy"))
    trans_times_true = safe_load(os.path.join(data_folder, "transition_times_true.npy"))
    trans_times_pred = safe_load(os.path.join(data_folder, "transition_times_pred.npy"))
    original_noise = safe_load(os.path.join(data_folder, "dl_traintime.npy"))
    separated_noise = safe_load(os.path.join(data_folder, "noise_separated.npy"))
    return train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise


# 创建复合图形的函数
def create_1d_composite_figure(train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise, save_name="1d_composite_plot.pdf"):
    # 图形参数设置
    plt.rcParams.update({
        'font.size': 15,   #12
        'axes.titlesize': 15,  # 14
        'axes.labelsize': 15,  # 14
        'font.family': 'Times New Roman',
        'axes.linewidth': 2,  # 1.5
    })


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    dt = 0.001

    ax1 = axes[0, 0]
    train = train_data.squeeze()[:10000]
    train_time = np.linspace(0, len(train) * dt, len(train))
    ax1.plot(train_time, train, color='salmon', label='Training Data')
    ax1.set(xlabel='Time', ylabel=r'$u_{t}$')
    ax1.set_ylim(-2, 2)
    ax1.legend(ncol=4, loc='upper right')
    ax1.text(-0.04, 1.1, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')

   
    ax2 = axes[0, 1]
    test = test_data.squeeze()[:10000]
    pred = pred_data.squeeze()[:10000]
    test_time = np.linspace(100, 100 + len(test) * dt, len(test))
    pred_time = np.linspace(100, 100 + len(pred) * dt, len(pred))
    ax2.plot(test_time, test, color='salmon', label='Test Data', alpha=0.6)
    ax2.plot(pred_time, pred, color='lightblue', label='Predicted Data')
    ax2.set(xlabel='Time', ylabel=r'$u_{t}$')
    ax2.set_ylim(-2, 2)
    ax2.legend(ncol=4, loc='upper right')
    ax2.text(-0.04, 1.1, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')


    ax3 = axes[1, 0]
    noise = separated_noise

    
    n, bins, _ = ax3.hist(noise, bins=400, density=True, color='lightblue', alpha=0.6, label='Separated Noise')

    
    alpha = 1.5
    beta = 0
    mu = 0
    sigma = dt ** (1 / alpha)
    x_levy = np.linspace(np.min(noise), np.max(noise), 10000)
    levy_pdf = levy_stable.pdf(x_levy, alpha, beta, loc=mu, scale=sigma)
    ax3.plot(x_levy, levy_pdf, color='salmon', linestyle='--', label='Theoretical Noise (Lévy)')

  
    y_max = max(np.max(n), np.max(levy_pdf))
    ax3.set_ylim(1e-3, y_max * 1.1)
    ax3.set_xlim(-0.6, 0.6)
    ax3.set(xlabel='Noise Amplitude', ylabel='PDF')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.text(-0.04, 1.1, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')

  
    ax4 = axes[1, 1]
    times_true = trans_times_true * dt
    times_pred = trans_times_pred * dt
    counts_true = int(trans_counts_true)
    counts_pred = int(trans_counts_pred)

  
    left = min(np.min(times_true), np.min(times_pred))
    right = max(np.max(times_true), np.max(times_pred))
    bins = np.arange(np.floor(left), np.ceil(right), 1)

    ax4.hist([times_true, times_pred], bins=bins,
             density=True, color=['salmon', 'lightblue'])
    ax4.set(xlabel='Time', ylabel='PDF')
    ax4.set_xlim(0, 25)
 
    legend_text = [
        f'Test: {counts_true} (μ={np.mean(times_true):.2f})',
        f'Pred: {counts_pred} (μ={np.mean(times_pred):.2f})'
    ]
    ax4.legend(legend_text, title='Transition Counts', loc='upper right')
    ax4.text(-0.04, 1.1, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')


    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    print(f"save to: {save_path}")
    plt.show()


if __name__ == "__main__":
    train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise = load_1d_data()
    create_1d_composite_figure(train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise)
    