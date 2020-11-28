
import logging
import numpy as np

"""
Disjoint LinUCB algorithm.
The variant was based on a brilliant material https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class DisjointLinUCB:

    def __init__(self, arm_id, dim, alpha, debug=True):
        """
        Class for one arm in Disjoint LinUCB contextual MAB algorithm
        :param arm_id: int, arm index
        :param dim:    int, a dimension of design matrix of context, number of features
        :param alpha:  float, hyperparameter in upper confidence bound (UCB) calculation

        """
        self.arm_id = arm_id
        self.dim = dim
        self.alpha = alpha

        # Initialize parameter for an arm
        # Init A as an identity matrix of dimension dim, A = D.T* D + I
        self.A = np.identity(self.dim)
        # Init b vector of dimension (dim, 1), b = D.T*y
        self.b = np.zeros((self.dim, 1))
        # Init theta vector in regression
        self.theta = None
        # Init inverse A matrix
        self.A_inv = None

        self.debug = debug

    def get_ucb(self, x):
        """
        Calculate ucb based on feature vector x
        :param x: array of features
        :return: float, ucb value
        """
        # Calculate theta vector for regression based on context
        # theta = A^-1 * b
        self.A_inv = np.linalg.pinv(self.A)
        self.theta = np.dot(self.A_inv, self.b)
        x = x.reshape((-1, 1))
        # ucb = mu + alpha*std
        #  mu = theta.T * x, std = sqrt(x.T * (D.T* D)^-1 * x)
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(self.A_inv, x)))
        return p

    def update_reward(self, reward, x):
        """
        Update reward
        :param reward, float or int, the value of a reward
        :param x: array of features
        :return: None
        """
        x = x.reshape((-1, 1))
        self.A += np.dot(x, x.T)
        self.b += reward*x


class LinUCBPolicy:
    """
    LinUCB with disjoint linear models for K-armed bandit
    """
    def __init__(self, arms, dim, alpha, seed=42, debug=False):
        """
        :param arms:  int, number of arms for a bandit to make
        :param dim:   int, dimension of feature matrix
        :param alpha: float, hyperparameter in upper confidence bound (UCB) calculation
        :param seed:  int, seed for random.seed
        """
        np.random.seed(seed)
        self.debug = debug
        self.arms = arms
        self.dim = dim
        self.alpha = alpha
        assert self.arms > 0
        self.linucb_arms = [DisjointLinUCB(arm_id=i, dim=self.dim, alpha=self.alpha) for i in range(self.arms)]
        logger.info(f'Created LinUCBPolicy with {self.arms} bandit arms.')

    def select_arm(self, x):

        # The shape of a feature vector is equal to the predefined dimension
        assert x.shape[0] == self.dim

        highest_ucb = -1
        candidate_arms = []
        candidates_ucbs = []
        for arm_id in range(self.arms):
            arm_ucb = self.linucb_arms[arm_id].get_ucb(x)

            if arm_ucb >= highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms.append(arm_id)
                candidates_ucbs.append(arm_ucb)

        chosen_arm = np.random.choice(candidate_arms)
        if self.debug:
            logger.debug(f'Candidate arms: {candidate_arms}\nCorresponding UCBs: {candidates_ucbs}')
        return chosen_arm


def simulate_recommendations(arms, dim, alpha, data_path, info=True):

    """
    :param arms:
    :param dim:
    :param alpha:
    :param data_path:
    :param info:
    :return: aligned_time_steps, cumulative_rewards, aligned_ctr, disjoint_linucb
    """

    """
    webscope-logs.txt data description:
    columns: 0    - timestamp;
             1-6  - user features;
             7    - article id that was selected uniformly at random in the system;
             8    - reward (click or no click);
             9-28 - articles that can be selected to the top of the site by a recommender system during the session;
    
    """
    disjoint_linucb = LinUCBPolicy(arms=arms, dim=dim, alpha=alpha)

    # Init trackers
    total_steps = 0
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []

    with open(data_path, 'r') as f:
        for line in f:
            # Extract and process logged data row
            line = line.split()
            timestamp = int(line[0])
            arm_list = [int(x)for x in line[9:29]]
            x = np.array([float(x) for x in line[1:7]])
            logged_arm = int(line[7])
            reward = int(line[8])

            assert arms == len(arm_list)

            # Find best arm in MAB
            chosen_arm_id = disjoint_linucb.select_arm(x)

            # Inspect requirements for offline evaluation: chosen arm (title) == logged title
            if arm_list[chosen_arm_id] == logged_arm:
                # Update arm information
                disjoint_linucb.linucb_arms[chosen_arm_id].update_reward(reward=reward, x=x)
                # Add data for ctr calculation
                aligned_time_steps += 1
                cumulative_rewards += reward
                aligned_ctr.append(cumulative_rewards/aligned_time_steps)

                if info:
                    logger.info(f'\nTotal steps: {total_steps}' +
                                f'\nAligned steps: {aligned_time_steps}' +
                                f'\nCumulative rewards: {cumulative_rewards}' +
                                f'\nAligned CTR: {aligned_ctr[-1]}\n\n')
            total_steps += 1

    return aligned_time_steps, cumulative_rewards, aligned_ctr, disjoint_linucb

# https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/notebooks/LinUCB_hybrid.ipynb
