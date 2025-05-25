import os
import numpy as np
import matplotlib.pyplot as plt

# 全局设置
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

def safe_load(file_path):
    try:
        data = np.load(file_path)
        print(f"✔ Successfully loaded {os.path.basename(file_path)}")
        return data
    except FileNotFoundError:
        print(f"❌ File {os.path.basename(file_path)} not found")
        return None

def load_data():
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    parent_dir = os.path.dirname(current_script_dir)
    data_folder = os.path.join(parent_dir, 'data')

    def load_and_print_shape(file_path):
        data = safe_load(file_path)
        if data is not None:
            print(f"Shape of {os.path.basename(file_path)}: {data.shape}")
        return data

    train_data = load_and_print_shape(os.path.join(data_folder, "original data.npy"))
    test_data = load_and_print_shape(os.path.join(data_folder, "test_data.npy"))
    pred_data = load_and_print_shape(os.path.join(data_folder, "predicted_data.npy"))
    trans_counts_true = load_and_print_shape(os.path.join(data_folder, "transition_counts_true.npy"))
    trans_counts_pred = load_and_print_shape(os.path.join(data_folder, "transition_counts_pred.npy"))
    trans_times_true = load_and_print_shape(os.path.join(data_folder, "transition_times_true.npy"))
    trans_times_pred = load_and_print_shape(os.path.join(data_folder, "transition_times_pred.npy"))
    original_noise = load_and_print_shape(os.path.join(data_folder, "dL_traintime.npy"))
    separated_noise = load_and_print_shape(os.path.join(data_folder, "noise_separated.npy"))

    return train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise

def plot_transition_histogram(trans_counts_true, trans_counts_pred,
                              trans_times_true, trans_times_pred,
                              dt=0.001, save_path="transition_histogram.png", save=True):
    times_true = trans_times_true * dt
    times_pred = trans_times_pred * dt

    mean_true = np.mean(times_true)
    mean_pred = np.mean(times_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    bins = np.linspace(min(times_true.min(), times_pred.min()),
                       max(times_true.max(), times_pred.max()),
                       40)
    color_true = '#F5A889'
    color_pred = '#ACD6EC'

    ax.hist(times_true, bins=bins, alpha=0.8,
            color=color_true, label=(
                f'True\n\n'
                f'Numbers={trans_counts_true[0]}\n\n'
                f'Mean time={mean_true:.2f}'
            ), density=True)
    ax.hist(times_pred, bins=bins, alpha=0.8,
            color=color_pred, label=(
                f'Predicted\n\n'
                f'Numbers={trans_counts_pred[0]}\n\n'
                f'Mean time={mean_pred:.2f}'
            ), density=True)

    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel('PDF', fontsize=18)
    ax.set_title('Transition Interval Distribution Comparison', fontsize=18)

    ax.legend(
        loc='upper right',
        frameon=True,
        title='Statistics:',
        title_fontsize=18,
        fontsize=18,
        borderpad=1,
        labelspacing=1.5,
        handleheight=2.5
    )

    plt.xlim(0, 25)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

def plot_time_series_comparison(train_data, test_data, pred_data,
                                dt=0.001, train_show_seconds=100,
                                test_show_seconds=10, save_path="ts_comparison.pdf", save=True):
    train_series = train_data.squeeze()
    test_series = test_data.squeeze()
    pred_series = pred_data.squeeze()

    train_show_points = int(train_show_seconds / dt)
    test_show_points = int(test_show_seconds / dt)

    train_segment = train_series[-train_show_points:]
    test_segment = test_series[:test_show_points]
    pred_segment = pred_series[:test_show_points]

    train_time = np.arange(-train_show_seconds, 0, dt)[:len(train_segment)]
    test_time = np.arange(0, test_show_seconds, dt)[:len(test_segment)]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_time, train_segment,
            color='#2c7bb6', lw=1.5, label='Training Data')
    ax.plot(test_time, test_segment,
            color='#2ca02c', lw=1.5, label='True Test Data')
    ax.plot(test_time, pred_segment,
            color='#d62728', lw=1.5, linestyle='--', label='Predicted Data')

    ax.axvline(0, color='gray', linestyle=':', lw=2, alpha=0.8,
               label='Train/Test Boundary')
    ax.annotate('', xy=(0.5, 0.95), xytext=(-0.5, 0.95),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax.text(0.25, 0.95, 'Transition Phase', transform=ax.transAxes,
            ha='center', va='center', fontsize=16, color='gray',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('Time (seconds)', fontsize=18)
    ax.set_ylabel('Value', fontsize=18)
    ax.set_title(f'Time Series Comparison\n'
                 f'[Last {train_show_seconds}s Training + First {test_show_seconds}s Test]',
                 fontsize=20, pad=20)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(left=train_time[0], right=test_time[-1])
    ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Transition histogram saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    save = False  # Set save switch: True for saving, False for displaying
    train_data, test_data, pred_data, trans_counts_true, trans_counts_pred, trans_times_true, trans_times_pred, original_noise, separated_noise = load_data()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    plot_transition_histogram(
        trans_counts_true=trans_counts_true,
        trans_counts_pred=trans_counts_pred,
        trans_times_true=trans_times_true,
        trans_times_pred=trans_times_pred,
        dt=0.001,
        save_path=os.path.join(current_script_dir, 'transition_histogram.pdf'),
        save=save
    )

    plot_time_series_comparison(
        train_data=train_data,
        test_data=test_data,
        pred_data=pred_data,
        dt=0.001,
        train_show_seconds=100,
        test_show_seconds=20,
        save_path=os.path.join(current_script_dir, 'time_series_comparison.pdf'),
        save=save
    )