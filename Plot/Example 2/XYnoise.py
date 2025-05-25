import numpy as np
import matplotlib.pyplot as plt
from LCload import load_2d_dataset
import os
def plot_noise_histograms():
    # Set plot parameters
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'font.family': 'Times New Roman',
        'axes.linewidth': 2.5,
    })

    # Load 2D limit cycle dataset
    dataset = load_2d_dataset()
    noise_x = dataset.noise_x
    noise_y = dataset.noise_y

    #  Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot X-direction noise histogram
    ax.hist(noise_x, bins=50, density=True, color='#148F28', alpha=0.6, label='Separated Noise (X-direction)')

    # Plot Y-direction noise histogram
    ax.hist(noise_y, bins=50, density=True, color='#EA71AE', alpha=0.6, label='Separated Noise (Y-direction)')

    # Set axis labels and title
    ax.set_xlabel('Noise Amplitude')
    ax.set_ylabel('PDF')
    ax.set_yscale('log')
    ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)

    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build save path
    save_path = os.path.join(script_dir, 'noise_histograms.pdf')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    plot_noise_histograms()