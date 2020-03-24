import pandas as pd
import numpy as np
from pylib.ann.cluster import CustomKMeans


if __name__ == '__main__':
	# Reading the SIFT data set
    X = np.fromfile('./data/SIFT/SIFT.dat', dtype=np.uint8).astype(np.float32)
    X = X.reshape((-1, 128))
    print(f'Descriptors format: {X.shape}')
	
	# Evaluate
    METHOD = 'exact'
    print(f'Method: {METHOD}')
    km = CustomKMeans(
    	n_centers=32000, 
    	method=METHOD,
    	nn_autotune=0.99,
    	nn_checks=100,
    	nn_kd_trees=64,
    	nn_km_branching=64,
    	nn_km_iter=20,
    	verbose=True
    )
    km.fit(X)
    print(km.time_report())
    df = pd.DataFrame(km.stats_)
    df = df.reset_index(drop=False)
    df.to_csv(f'./data/sift_comparison_{km.session_id}.csv', index=False)
