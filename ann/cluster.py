import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist
from time import time
from pyflann import FLANN
from datetime import datetime
from pathlib import Path


class CustomKMeans(object):
    
    def __init__(self, 
                 n_centers: int = 2,
                 method: str = 'kmeans',
                 max_iter: int = 30,
                 max_iter_no_progress: int = 10, 
                 tol_progress: np.float = 1e-3, 
                 random_state: Optional[int] = 17,
                 nn_km_branching: int = 32,  # branching for k-means tree
                 nn_km_iter: int = 20, # number of iterations per k-means step
                 nn_kd_trees: int = 32,  # number of randomized trees to use
                 nn_checks: int = 75,  # number of leaves to check in the search
                 nn_autotune: np.float = -1,  # auto-tuning of nn parameters
                 save_log: bool = True,
                 verbose: bool = False):
        
        assert method in {'kmeans', 'kdtree', 'exact'}
        
        # FLANN hyperparameters
        # The whole list of FLANN parameters is available here:
        # https://github.com/mariusmuja/flann/blob/master/src/cpp/flann/flann.h
        self.flann = None
        self.params = {
            'kdtree': {
                'algorithm': 'kdtree',
                'num_neighbors': 1,
                'trees': nn_kd_trees, 
                'checks': nn_checks,
                'target_precision': nn_autotune
            },
            'kmeans': {
                'algorithm': 'kmeans',
                'num_neighbors': 1,
                'branching': nn_km_branching,  
                'iterations': nn_km_iter, 
                'checks': nn_checks,
                'target_precision': nn_autotune
            },
            'exact':
            {
                'algorithm': 'exact'
            }
        }
        
        # Set target precision in case of autotuning
        if method in {'kmeans', 'kdtree'}:
            self.flann = FLANN()
        
        self.n_centers = n_centers
        self.method = method
        self.max_iter = max_iter
        self.max_iter_no_progress = max_iter_no_progress
        self.tol_progress = tol_progress
        self.random_state = random_state
        self.verbose = verbose
        
        self._dim = None
        self._n_samples = None
        self.centers_ = None
        self.labels_ = None
        self.stats_ = None
        self.session_id = f'n_centers_{self.n_centers}-' + \
            '-'.join(
            [f'{param}_{val}' for param, val 
             in self.params[self.method].items()]
        )
        self.log_ = []
        self.save_log = save_log
        
    def __reset_stats(self):
        self.stats_ = {
            'measure': [],  # the cost function
            'evaluation': [],  # time to evaluate labels
            'assignment': [],  # time to re-assign labels
            'n_centers': self.n_centers,
            **self.params[self.method]
        }
    
    def __update_stats(self, 
                       measure: np.float, 
                       time_eval: np.float, 
                       time_assign: np.float):
        self.stats_['measure'].append(measure)
        self.stats_['evaluation'].append(time_eval)
        self.stats_['assignment'].append(time_assign)
        
    def fit(self, data: np.ndarray) -> None:
        self._n_samples, self._dim = data.shape
        
        np.random.seed(self.random_state)
        
        # Initialize centers (we use uniform random initialization)
        self.centers_ = np.random.uniform(
            np.min(data), np.max(data), 
            size=self.n_centers * self._dim
        ).reshape(self.n_centers, self._dim).astype(data.dtype)
        
        # Start fitting
        self.__reset_stats()
        self.log_.append(
            f'Session id: {self.session_id} '
            f'at {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}'
        )
        progress = self.max_iter_no_progress
        for it in range(self.max_iter):
            
            tic_it = time()

            # Evaluate labels and squared distances
            if self.method in {'kmeans', 'kdtree'}:
                self.labels_, sqdist = self.flann.nn(
                    self.centers_, data, **self.params[self.method])
            else:
                self.labels_ = np.ndarray((self._n_samples,), dtype=np.int)
                sqdist = np.ndarray((self._n_samples,))
                for p in range(self._n_samples):
                    sqpdist = cdist(data[p].reshape(1, -1), self.centers_, 
                                    metric='sqeuclidean')
                    self.labels_[p] = sqpdist.argmin(axis=1)[0]
                    sqdist[p] = sqpdist[0, self.labels_[p]]
                # Vectrorised approach
                # sqpdist = cdist(data, self.centers_, metric='sqeuclidean')
                # self.labels_ = sqpdist.argmin(axis=1)
                # sqdist = sqpdist[np.arange(self._n_samples), self.labels_]
            toc = time()
            t1 = toc - tic_it
            
            # Update centers
            tic = time()
            for label in range(self.n_centers):
                idx = np.where(self.labels_ == label)[0]
                if len(idx):
                    self.centers_[label] = data[idx].mean(axis=0)
                else:
                    # If the cluster is empty, move its center anyway
                    self.centers_[label] = self.centers_.mean(axis=0)
            toc_it = time()
            t2 = toc_it - tic
            
            # Print progress
            self.log_.append(
                f'\t--> iteration {it} '
                f'has been finished -- {toc_it - tic_it:.3e}s'
            )
            if self.verbose:
                print(self.log_[-1])
                    
            # Check convergence
            p = np.bincount(self.labels_, weights=sqdist).sum() / data.shape[0]
            if it > 1 and self.stats_['measure'][-2] - \
                    self.stats_['measure'][-1] < self.tol_progress:
                progress -= 1
                if not progress:
                    self.__update_stats(p, t1, t2)
                    print(f'\nIteration {it}: '
                          f'no progress during '
                          f'last {self.max_iter_no_progress} iterations')
                    break
            else:
                self.__update_stats(p, t1, t2)
                progress = self.max_iter_no_progress
        
        # Save the log
        self.log_.append(self.time_report())
        if self.save_log:
            path = Path(f'./logs/{self.__class__.__name__}/').resolve()
            path.mkdir(parents=True, exist_ok=True)
            with open(f'{str(path)}/{self.session_id}.txt', 'w') as file:
                for line in self.log_:
                    file.write(f'{line}\n')
        
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        
        self.fit(data)
        
        return self.labels_
    
    def time_report(self) -> str:
        if self.stats_ is None:
            rep = 'No statistics are available.'
        else:
            t_eval = np.asarray(self.stats_['evaluation'])
            t_assign = np.asarray(self.stats_['assignment'])
            rep = f'\nEvaluation time per iteration:\n\tAVG. = {t_eval.mean()}s\n\tSTD. = {t_eval.std()}s\nAssignment time per iteration:\n\tAVG. = {t_assign.mean()}s\n\tSTD. = {t_assign.std()}s'
        return rep


if __name__=='__main__':
	pass
