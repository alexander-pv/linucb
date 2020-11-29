
import os
from src import linucb

if __name__ == '__main__':

    LOGS_PATH = os.path.join('..', 'dataset', 'webscope-logs.txt')
    ARTICLES_PATH = os.path.join('..', 'dataset', 'webscope-articles.txt')
    aligned_time_steps, cumulative_rewards, aligned_ctr, disjoint_linucb = \
        linucb.simulate_recommendations(arms=20, user_dim=6, article_dim=6,
                                        lin_type='hybrid', alpha=0.5,
                                        logs_path=LOGS_PATH, articles_path=ARTICLES_PATH)
