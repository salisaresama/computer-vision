import pandas as pd
import numpy as np
import os
from cluster import CustomKMeans


if __name__ == '__main__':
    print(os.listdir('.'))
    print(os.listdir('../data'))
    os.system('tar -xf ../data/sift.zip -C ../data/')
    # Reading the SIFT data set
    X = np.fromfile('../data/SIFT.dat', dtype=np.uint8).astype(np.float32)
    X = X.reshape((-1, 128))
    print(f'Descriptors format: {X.shape}')

    # Evaluate
    METHOD = 'exact-gpu'
    print(f'Method: {METHOD}')
    km = CustomKMeans(
        n_centers=10,
        max_iter=5,
        method=METHOD,
        verbose=True
    )
    km.fit(X)
    print(km.time_report())
    df = pd.DataFrame(km.stats_)
    df = df.reset_index(drop=False)
    df.to_csv(f'../output/oxford_comparison_{km.session_id}.csv', index=False)
