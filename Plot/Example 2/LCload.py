# data_loader.py
import os
import numpy as np
from typing import NamedTuple

class Dataset2D(NamedTuple):
    """2D dataset container"""
    train: np.ndarray     # Training data (2, N)
    test: np.ndarray      # Test data (2, M)
    pred: np.ndarray      # Test data (2, M)
    counts_true: int      # True transition counts
    counts_pred: int      # Predicted transition counts
    times_true: np.ndarray  # True transition times (K,)
    times_pred: np.ndarray  # Predicted transition times (L,)
    noise_x: np.ndarray   # X-direction noise (N-1,)
    noise_y: np.ndarray   # Y-direction noise (N-1,)

def load_2d_dataset(base_dir=None) -> Dataset2D:
    """load the 2D dataset
    
    Args:
        base_dir: Optional root directory path, auto-detected by default
        
    Returns:
        Dataset2D: Structured dataset collection
    """
    # 路径解析
    if not base_dir:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(
        os.path.dirname(current_dir),  # Prevent redundant parent directory access
        '..',  # 
        'data',
        '2D limit cycle'  # 
    )
   
    data_dir = os.path.normpath(data_dir)
    
    # file map
    file_map = {
        'train': '2D limit cycle train_data.npy',
        'test': '2D limit cycle test_data.npy',
        'pred': '2D limit cycle predicted data.npy',
        'noise_x': '2D limit cycle noise x direction .npy',
        'noise_y': '2D limit cycle noise y direction .npy',
        'counts_true': '2D limit cycle transition_counts_true.npy',
        'counts_pred': '2D limit cycle transition_counts_pred.npy',
        'times_true': '2D limit cycle transition_times_true.npy',
        'times_pred': '2D limit cycle transition_times_pred.npy'
    }
    
    # load data
    loaded = {}
    for key, filename in file_map.items():
        path = os.path.join(data_dir, filename)
        try:
            data = np.load(path)
            loaded[key] = data.squeeze() if key.startswith('counts') else data
        except Exception as e:
            raise RuntimeError(f"load {filename} failed: {str(e)}") from None
    
    # validate
    _validate_shapes(
        train=loaded['train'],
        test=loaded['test'],
        pred=loaded['pred'],
        noise_x=loaded['noise_x'],
        noise_y=loaded['noise_y']
    )
    
    return Dataset2D(
        train=loaded['train'],
        test=loaded['test'],
        pred=loaded['pred'],
        counts_true=int(loaded['counts_true']),
        counts_pred=int(loaded['counts_pred']),
        times_true=loaded['times_true'],
        times_pred=loaded['times_pred'],
        noise_x=loaded['noise_x'],
        noise_y=loaded['noise_y']
    )

def _validate_shapes(**data_arrays):
    """internal function to validate data shapes"""
    shape_rules = {
        'train': (2, None),
        'test': (2, None),
        'pred': (2, None),
        'noise_x': (None,),
        'noise_y': (None,)
    }
    
    for name, array in data_arrays.items():
        expected_dim = len(shape_rules[name])
        if array.ndim != expected_dim:
            raise ValueError(
                f"{name} wrong dimension: expected: {expected_dim}D, true {array.ndim}D"
            )


if __name__ == "__main__":
    data = load_2d_dataset()
    print(data)