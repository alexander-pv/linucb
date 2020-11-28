
import os
from src import linucb

if __name__ == '__main__':

    DATA_PATH = os.path.join('..', 'dataset', 'webscope-logs.txt')
    aligned_time_steps, cumulative_rewards, aligned_ctr, disjoint_linucb = \
        linucb.simulate_recommendations(arms=20, dim=6, alpha=0.5, data_path=DATA_PATH)
