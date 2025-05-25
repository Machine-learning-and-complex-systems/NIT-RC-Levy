import numpy as np
import sdeint
import matplotlib.pyplot as plt
import os

def simulate_system(params=None, deterministic=False):
    """Perform SDE simulation and return results
    Args:
        params (dict): Optional parameter override dictionary
    Returns:
        dict: Dictionary containing results and parameters
    """
    # Default parameter settings
    default_params = {
        'mu': 0.2,
        'gamma_right': 1,
        'k': 5,
        'x0_transition': -0.0,
        'sigma': 0.8,
        'threshold': 0.0,
        'total_time': 100.0,
        'dt': 0.01,
        'num_trajectories': 1,
        'seed': None
    }

    # Merge parameters
    if params:
        default_params.update(params)
    p = default_params

    if deterministic:
        p['sigma'] = 0
        p['total_time'] = 100

    # 
    if p['seed'] is not None:
        np.random.seed(p['seed'])

    # Define system functions
    def U(x):
        return x ** 4 / 4 - (1 - 0.0) * x ** 2 / 2

    def dU_dx(x):
        return x ** 3 - (1 - 0.0) * x

    def sigma_trans(x):
        return 1 / (1 + np.exp(-p['k'] * (x - p['x0_transition'])))

    def gamma(x):
        return p['gamma_right'] * sigma_trans(x) + p['mu'] * (x ** 2 - 1) * (1 - sigma_trans(x))

    def f(X, t):
        return np.array([X[1], -dU_dx(X[0]) - gamma(X[0]) * X[1]])

    def g(X, t):
        return np.array([[0.0], [p['sigma']]])

    # Time grid
    tspan = np.linspace(0, p['total_time'], int(p['total_time'] / p['dt']))

    # Store results
    results = []
    for _ in range(p['num_trajectories']):
        X0 = np.array([-1.5 + 0.3 * np.random.randn(),
                       0.0 + 0.5 * np.random.randn()])
        result = sdeint.stratHeun(f, g, X0, tspan)
        results.append({
            't': tspan,
            'x': result[:, 0],
            'y': result[:, 1],
            'cross_time': None
        })

        # Detect transition time
        cross_idx = np.where(result[:, 0] > p['threshold'])[0]
        if len(cross_idx) > 0:
            results[-1]['cross_time'] = tspan[cross_idx[0]]

    return {'results': results, 'params': p}


def plot_single_phase_space():
    # Set plot parameters with increased font size
    plt.rcParams.update({
        'font.size': 21,
        'axes.titlesize': 21,
        'axes.labelsize': 21,
        'font.family': 'Times New Roman',
        'axes.linewidth': 2.5,
    })
    # Generate deterministic trajectories
    deterministic_params = {
        'num_trajectories': 5,  # Increased trajectory count for convergence demonstration
        'total_time': 100,
        'sigma': 0
    }
    deterministic_data = simulate_system(deterministic_params, deterministic=True)

 
    fig, ax = plt.subplots(figsize=(8, 6))

    
    for i, traj in enumerate(deterministic_data['results']):
        sc = ax.scatter(traj['x'], traj['y'], s=1, c=traj['t'], cmap='viridis', label=f'Trajectory {i + 1}')

        # marking the starting point
        start_x, start_y = traj['x'][0], traj['y'][0]
        ax.scatter(start_x, start_y, s=50, c='salmon', marker='o')
        ax.text(start_x, start_y, f'Start {i + 1}', color='black', fontsize=18)




    fig.colorbar(sc, ax=ax, label='Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title("2D Scatter Plot")

  
    # ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)
    # # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Bulid save path
    save_path = os.path.join(script_dir, 'deterministic.pdf')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    plot_single_phase_space()
    