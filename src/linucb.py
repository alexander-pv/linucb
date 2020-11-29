
import logging
import numpy as np

"""
Disjoint and Hybrid LinUCB algorithms.
The variant was based on a brilliant material:
 https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
 https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/
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
        # Init b vector of dimension (dim, 1), b = D.T*c
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


class HybridLinUCB:

    def __init__(self, arm_id, user_dim, article_dim, alpha, debug=True):
        self.arm_id = arm_id
        self.user_dim = user_dim
        self.article_dim = article_dim
        self.alpha = alpha
        self.debug = debug

        # D.T * D matrix
        self.A = np.identity(self.user_dim)
        # Init b vector of dimension (dim, 1), b = D.T*c
        self.b = np.zeros((self.user_dim, 1))
        # B = D.T * c in ridge regression
        self.B = np.zeros((self.user_dim, self.article_dim*self.user_dim))

        self.arm_features = None
        self.A_shared_inv = None
        self.b_shared = None
        self.theta = None
        self.beta_hat = None
        self.z_array = None

    def store_arm_features(self, arm_features):
        self.arm_features = arm_features

    def get_ucb(self, x):
        # Outer product based on user features and article (arm) features
        x = x.reshape((-1, 1))
        z_array = np.outer(self.arm_features, x).reshape(-1, 1)
        A_inv = np.linalg.pinv(self.A)

        # theta parameter of an arm
        self.theta = np.dot(A_inv, self.b - np.dot(self.B, self.beta_hat))
        # standard deviation of an arm
        s = np.dot(z_array.T, np.dot(self.A_shared_inv, z_array)) - \
            2 * np.dot(z_array.T, np.dot(self.A_shared_inv, np.dot(self.B.T, np.dot(A_inv, x)))) + \
            np.dot(x.T, np.dot(A_inv, x)) + \
            np.dot(x.T, np.dot(A_inv, np.dot(self.B, np.dot(self.A_shared_inv, np.dot(self.B.T, np.dot(A_inv, x))))))

        p = np.dot(z_array.T, self.beta_hat) + np.dot(x.T, self.theta) + self.alpha*np.sqrt(s)
        return p

    def update_reward(self, reward, x):

        self.z_array = np.outer(self.arm_features, x).reshape(-1, 1)
        x = x.reshape(-1, 1)
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        # Update B which is (d * k) matrix.
        self.B += np.dot(x, self.z_array.T)
        self.b += reward * x


class LinUCBPolicy:
    """
    LinUCB with disjoint linear models for K-armed bandit
    """
    def __init__(self, arms, user_dim, article_dim, alpha, lin_type, seed=42, debug=False):
        """
        :param arms:  int, number of arms for a bandit to make
        :param user_dim:    int, dimension of user feature matrix
        :param article_dim: int, dimension of article (arm) feature matrix
        :param alpha: float, hyperparameter in upper confidence bound (UCB) calculation
        :param lin_type:  string, 'disjoint' or 'hybrid'
        :param seed:  int, seed for random.seed
        :param debug: bool
        """
        np.random.seed(seed)
        self.lin_type = lin_type
        self.debug = debug
        self.arms = arms
        self.user_dim = user_dim
        self.article_dim = article_dim
        self.alpha = alpha

        assert self.arms > 0
        assert self.lin_type in ['disjoint', 'hybrid']

        if self.lin_type == 'disjoint':
            self.linucb_arms = [DisjointLinUCB(arm_id=i, dim=self.user_dim, alpha=self.alpha) for i in range(self.arms)]
        elif self.lin_type == 'hybrid':
            self.linucb_arms = [HybridLinUCB(arm_id=i, user_dim=self.user_dim, article_dim=self.article_dim,
                                             alpha=self.alpha) for i in range(self.arms)]
            self.A_shared = np.identity(self.article_dim*self.article_dim)
            self.b_shared = np.zeros((self.user_dim*self.article_dim, 1))
        else:
            raise NotImplementedError

        logger.info(f'Created LinUCBPolicy ({self.lin_type} linear models) with {self.arms} bandit arms.')

    def select_arm(self, x):

        # The shape of a feature vector is equal to the predefined dimension
        assert x.shape[0] == self.user_dim

        chosen_arm = int()
        highest_ucb = -1
        candidate_arms = []
        candidates_ucbs = []

        if self.lin_type == 'disjoint':
            for arm_id in range(self.arms):
                arm_ucb = self.linucb_arms[arm_id].get_ucb(x)

                if arm_ucb >= highest_ucb:
                    highest_ucb = arm_ucb
                    candidate_arms.append(arm_id)
                    candidates_ucbs.append(arm_ucb)

            chosen_arm = np.random.choice(candidate_arms)

        if self.lin_type == 'hybrid':
            A_shared_inv = np.linalg.pinv(self.A_shared)
            beta_hat = np.dot(A_shared_inv, self.b_shared)

            for arm_id in range(self.arms):
                self.linucb_arms[arm_id].A_shared_inv = A_shared_inv
                self.linucb_arms[arm_id].b_shared = self.b_shared
                self.linucb_arms[arm_id].beta_hat = beta_hat
                arm_ucb = self.linucb_arms[arm_id].get_ucb(x)

                if arm_ucb >= highest_ucb:
                    highest_ucb = arm_ucb
                    candidate_arms.append(arm_id)
                    candidates_ucbs.append(arm_ucb)
            chosen_arm = np.random.choice(candidate_arms)

        if self.debug:
            logger.debug(f'Candidate arms: {candidate_arms}\nCorresponding UCBs: {candidates_ucbs}')
        return chosen_arm


def get_essential_articles_features(logs_path, articles_path):
    """
    Collect essential articles (arms) features
    :param logs_path: str
    :param articles_path: str
    :return: dict with arm features, dict with arm id and article mapping
    """

    logs = open(logs_path, 'r')
    log_line = logs.readline().split()
    logs.close()

    arm_list = [int(x) for x in log_line[9:29]]
    articles_features = dict()
    arm_id_mapping = dict()  # {arm_id: article_id}
    with open(articles_path, 'r') as f:
        for line in f:
            article_id = int(line.split()[0])
            if article_id in arm_list:
                articles_features[article_id] = np.array([float(x) for x in line.split()[1:]])
                arm_id_mapping[arm_list.index(article_id)] = article_id
    return articles_features, arm_id_mapping


def simulate_recommendations(arms, user_dim, article_dim, lin_type, alpha, logs_path, articles_path, info=True):

    """
    :param arms:
    :param user_dim:
    :param article_dim:
    :param lin_type:
    :param alpha:
    :param logs_path:
    :param articles_path:
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
    linucb_instance = LinUCBPolicy(arms=arms, user_dim=user_dim, article_dim=article_dim,
                                   lin_type=lin_type, alpha=alpha)
    articles_features,  arm_id_mapping = get_essential_articles_features(logs_path=logs_path,
                                                                         articles_path=articles_path)
    if lin_type == 'hybrid':
        for i, arm in enumerate(linucb_instance.linucb_arms):
            arm.store_arm_features(articles_features[arm_id_mapping[i]])

    # Init trackers
    total_steps = 0
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []

    with open(logs_path, 'r') as f:
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
            chosen_arm_id = linucb_instance.select_arm(x)

            # Inspect requirements for offline evaluation: chosen arm (title) == logged title
            if arm_list[chosen_arm_id] == logged_arm:
                # Update arm information
                if lin_type == 'disjoint':
                    linucb_instance.linucb_arms[chosen_arm_id].update_reward(reward=reward, x=x)
                elif lin_type == 'hybrid':

                    # Step 1
                    chosen_arm_A = linucb_instance.linucb_arms[chosen_arm_id].A
                    chosen_arm_B = linucb_instance.linucb_arms[chosen_arm_id].B
                    chosen_arm_b = linucb_instance.linucb_arms[chosen_arm_id].b
                    chosen_arm_A_inv = np.linalg.pinv(chosen_arm_A)
                    linucb_instance.A_shared += np.dot(chosen_arm_B.T, np.dot(chosen_arm_A_inv, chosen_arm_B))
                    linucb_instance.b_shared += np.dot(chosen_arm_B.T, np.dot(chosen_arm_A_inv, chosen_arm_b))

                    # Step 2
                    linucb_instance.linucb_arms[chosen_arm_id].update_reward(reward=reward, x=x)

                    # Step 3
                    chosen_arm_A = linucb_instance.linucb_arms[chosen_arm_id].A
                    chosen_arm_B = linucb_instance.linucb_arms[chosen_arm_id].B
                    chosen_arm_b = linucb_instance.linucb_arms[chosen_arm_id].b
                    chosen_arm_A_inv = np.linalg.pinv(chosen_arm_A)

                    z_array = linucb_instance.linucb_arms[chosen_arm_id].z_array
                    linucb_instance.A_shared += np.dot(z_array, z_array.T) - \
                                                np.dot(chosen_arm_B.T, np.dot(chosen_arm_A_inv, chosen_arm_B))
                    linucb_instance.b_shared += reward * z_array - \
                                                np.dot(chosen_arm_B.T, np.dot(chosen_arm_A_inv, chosen_arm_b))

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

    return aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_instance
