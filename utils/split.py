import numpy as np

def random_split(x, y, N, seed=None):

    # Init seed
    if seed is not None:
        np.random.seed(seed)

    # Procude and shuffle indexes
    idx = np.arange(len(y))
    np.random.shuffle(idx, )

    # Create partitions
    x_sampled, y_ssampled, x_other, y_other = x[idx[:N], :], y[idx[:N]], x[idx[N:], :], y[idx[N:]]
    return x_sampled, y_ssampled, x_other, y_other


def balance_split(x, y, N, p=None, seed=None, allow_missing_class=False):


    # Create sampling number
    correct_n = 0
    split_values = []
    for val in list(np.unique(y)):
        N_y = np.floor(N*p[val])
        resd = N*p[val] - N_y
        split_value = int(N_y.item() + np.random.choice([0,1], p=[1-resd, resd]).item())
        if not allow_missing_class:
            if np.max(split_value) < 1:
                split_value = np.max([1, split_value])
                correct_n += 1
        split_values.append(split_value)

    # Retrieve indexes
    sampled_idx, other_idx = [], []
    for val in list(np.unique(y)):
        idx = np.where(y == val)[0]
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx, )
        sampled_idx.extend(idx[:split_values[val]])
        other_idx.extend(idx[split_value:])

    # Create partitions
    x_sampled, y_sampled, x_other, y_other = x[sampled_idx, :], y[sampled_idx], x[other_idx, :], y[other_idx]
    return x_sampled, y_sampled, x_other, y_other