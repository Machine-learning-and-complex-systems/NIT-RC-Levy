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
    # Default parameters
    default_params = {
        'mu': 0.2,
        'gamma_right': 1,
        'k': 5,
        'x0_transition': -0.0,
        'sigma': 0.8,
        'threshold': 0.0,
        'total_time': 1100.0,
        'dt': 0.01,
        'num_trajectories': 1,
        'seed': None
    }
    
   # Default parameters
    if params:
        default_params.update(params)
    p = default_params
    

    if deterministic:
        p['sigma'] = 0
        p['total_time'] = 100


    
    if p['seed'] is not None:
        np.random.seed(p['seed'])
    
    
    def U(x): 
        return x**4/4 - (1 - 0.0)*x**2/2
    def dU_dx(x): 
        return x**3 - (1 - 0.0)*x 
    def sigma_trans(x): 
        return 1 / (1 + np.exp(-p['k']*(x - p['x0_transition'])))
    def gamma(x): 
        return p['gamma_right']*sigma_trans(x) + p['mu']*(x**2 - 1)*(1 - sigma_trans(x))
    def f(X, t): 
        
        return np.array([X[1], -dU_dx(X[0]) - gamma(X[0])*X[1]])
    
    def g(X, t): return np.array([[0.0], [p['sigma']]])

    
    # time mesh
    tspan = np.linspace(0, p['total_time'], int(p['total_time']/p['dt']))
    
    # storge for results
    results = []
    for _ in range(p['num_trajectories']):
        X0 = np.array([-1.5 + 0.3*np.random.randn(),
                      0.0 + 0.5*np.random.randn()])
        result = sdeint.stratHeun(f, g, X0, tspan)
        results.append({
            't': tspan,
            'x': result[:,0],
            'y': result[:,1],
            'cross_time': None
        })
        
        # no use of cross_time
        cross_idx = np.where(result[:,0] > p['threshold'])[0]
        if len(cross_idx) > 0:
            results[-1]['cross_time'] = tspan[cross_idx[0]]
    
    return {'results': results, 'params': p}

def plot_results(data, show=True):
    """Visualize simulation results
    Args:
        data (dict): Return value from simulate_system
        show (bool): Whether to display immediately
    """
    p = data['params']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                  gridspec_kw={'width_ratios': [1, 2]})
    
    # phase space
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    ax1.contour(x_vals, y_vals, 
               [[p['mu']*(x**2-1) + 0.5*y**2 for y in y_vals] for x in x_vals],
               levels=15, cmap='Greys', alpha=0.2)
    
    # Plot all trajectories
    for i, traj in enumerate(data['results']):
        color = colors[i % len(colors)]
        
        
        ax1.plot(traj['x'], traj['y'], color=color, lw=1.5, alpha=0.8)
        if traj['cross_time'] is not None:
            cross_idx = np.argmin(np.abs(traj['t'] - traj['cross_time']))
            ax1.scatter(traj['x'][cross_idx], traj['y'][cross_idx],
                       color=color, marker='*', s=120, edgecolor='black')
        
        # time series
        ax2.plot(traj['t'], traj['x'], color=color, lw=1.2, alpha=0.7,
                label=f'Traj {i+1}')
        if traj['cross_time'] is not None:
            ax2.axvline(traj['cross_time'], color=color, ls='--', alpha=0.7)
    
    # Figure annotations
    ax1.set(xlim=(-2, 2), ylim=(-2, 2),
           xlabel='Position (x)', ylabel='Momentum (y)',
           title='Phase Space')
    ax1.axvline(p['x0_transition'], color='purple', ls=':', lw=2)
    
    ax2.set(xlim=(0, p['total_time']), ylim=(-2, 2),
           xlabel='Time', ylabel='Position (x)',
           title='Time Series')
    ax2.legend()
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig



def save_simulation_data(data, filename, parent_dir=None):
    """Save only trajectory data (x, y, t)"""
    p = data['params']
  
    param_str = f"mu_{p['mu']}_gamma_{p['gamma_right']}_sigma_{p['sigma']}"
    new_filename = f"{filename}_{param_str}.npz"

    # path
    if parent_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
    
    save_dir = os.path.join(parent_dir, 'data', '2D limit cycle')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, new_filename)

    # Extract trajectory data
    x = np.array([t['x'] for t in data['results']])  # shape: (n_traj, n_points)
    y = np.array([t['y'] for t in data['results']])  # shape: (n_traj, n_points)
    t = data['results'][0]['t']                      # shape: (n_points,)

    # Save as compressed .npz file
    np.savez_compressed(save_path, x=x, y=y, t=t)
    print(f"save to：{save_path}")

# 加载验证方法
def load_trajectories(filepath):
    """Load pure trajectory data"""
    data = np.load(filepath)
    return {
        'x': data['x'],
        'y': data['y'],
        't': data['t']
    }


if __name__ == "__main__":
    # save data
    save = False
    custom_params = {'num_trajectories': 1}
    simulation_data = simulate_system(custom_params)
    p = simulation_data['params']
    param_str = f"mu_{p['mu']}_gamma_{p['gamma_right']}_sigma_{p['sigma']}"
    filename = f"trajectories_only_{param_str}.npz"

    if save:
        save_simulation_data(simulation_data, "trajectories_only")
    


    # Plot switch (True to enable, False to disable)
    plot_enabled = True
    deterministic = True
    if plot_enabled:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        save_dir = os.path.join(parent_dir, 'data', '2D limit cycle')
        load_path = os.path.join(save_dir, filename)
        if save:
            loaded = load_trajectories(load_path)
            print("data dimension validation:")   
            print(f"x shape: {loaded['x'].shape}")  # 
            print(f"y shape: {loaded['y'].shape}")  # 
            print(f"t shape: {loaded['t'].shape}")  # 
       
        if deterministic:
            # Gnerate deterministic trajectory
            deterministic_params = custom_params.copy()
            deterministic_params['num_trajectories'] = 5  # add more trajectories for convergence
            deterministic_data = simulate_system(deterministic_params, deterministic=True)

             # Time series subplot - x component
            plt.subplot(3, 1, 1)
            for i, traj in enumerate(deterministic_data['results']):
                plt.plot(traj['t'], traj['x'], label=f'x_{i} (deterministic)')
            plt.legend()
            plt.title("Time Series of x (Deterministic)")

            # Time series subplot - y component
            plt.subplot(3, 1, 2)
            for i, traj in enumerate(deterministic_data['results']):
                plt.plot(traj['t'], traj['y'], label=f'y_{i} (deterministic)')
            plt.legend()
            plt.title("Time Series of y (Deterministic)")

            # New scatter subplot - deterministic trajectories
            plt.subplot(3, 1, 3)
            for i, traj in enumerate(deterministic_data['results']):
                plt.scatter(traj['x'], traj['y'], s=1, c=traj['t'], cmap='viridis')
            plt.colorbar(label='Time')
            plt.xlabel('Position (x)')
            plt.ylabel('Momentum (y)')
            plt.title("Phase Space Scatter (Deterministic)")

        else:
            # plot non-deterministic trajectory
            plt.subplot(3, 1, 1)
            plt.plot(loaded['t'], loaded['x'][0], label='x')
            plt.legend()

            
            plt.subplot(3, 1, 2)
            plt.plot(loaded['t'], loaded['y'][0], label='y')
            plt.subplot(3, 1, 2)
            plt.plot(loaded['t'], loaded['y'][0], label='y')
            plt.legend()
            plt.title("Time Series")

           
            plt.subplot(3, 1, 3)
            plt.scatter(loaded['x'][0], loaded['y'][0], s=1, c=loaded['t'], cmap='viridis')
            plt.colorbar(label='Time')
            plt.xlabel('Position (x)')
            plt.ylabel('Momentum (y)')
            plt.title("Phase Space Scatter")

        plt.tight_layout()
  


