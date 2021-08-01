from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.preprocessing import MinMaxScaler

columns = ['user', 'item', 'score']
train_data = pd.read_csv('dataset/train_user_item_score.txt', header=None, names=columns).drop_duplicates(columns[0:1])
validation_data = pd.read_csv('dataset/validation_user_item_score.txt', header=None, names=columns).drop_duplicates(
    columns[0:1])
users_items_train_matrix = train_data.pivot(index='user', columns='item', values='score')

users_graph = pd.read_csv('dataset/users_connections.txt', header=None, names=['user1', 'user2']).drop_duplicates(
    ['user1', 'user2'])

n_users = users_items_train_matrix.shape[0]
n_items = users_items_train_matrix.shape[1]
R = users_items_train_matrix.fillna(0).values

users_adjacency_list = defaultdict(list)

for _, connection in users_graph.iterrows():
    if connection['user1'] < n_users and connection['user2'] < n_users:
        users_adjacency_list[connection['user1']].append(connection['user2'])

n_epochs = 20
n_latent = 2


class VariationalMeanField:
    def __init__(self, sigma=1, alpha_U=1, beta_U=1, alpha_V=1, beta_V=1):
        self.sigma = sigma
        self.alpha_U = alpha_U
        self.beta_U = beta_U
        self.alpha_V = alpha_V
        self.beta_V = beta_V

        self.sigma_U: float = 1 / gamma.rvs(self.alpha_U, self.beta_U)
        self.sigma_V: float = 1 / gamma.rvs(self.alpha_V, self.beta_V)

        self.sigma_U = 1
        self.sigma_V = 1

        self.U = np.random.normal(0.0, self.sigma_U, (n_latent, n_users))
        self.V = np.random.normal(0.0, self.sigma_V, (n_latent, n_items))

        self.mean_U = np.random.normal(0.0, self.sigma_U, (n_latent, n_users))
        self.covs_U = [self.sigma_U * np.identity(n_latent) for _ in range(n_users)]

        self.mean_V = np.random.normal(0.0, self.sigma_V, (n_latent, n_items))
        self.covs_V = [self.sigma_V * np.identity(n_latent) for _ in range(n_items)]

        self.W = np.zeros((n_users, n_users))
        self.mean_W = np.zeros((n_users, n_users))
        self.sigmas_W = np.zeros((n_users, n_users))

    @staticmethod
    def _neighbors_R(user, item) -> np.array:
        neighbors = users_adjacency_list[user]
        return np.array([R[neighbor, item] for neighbor in neighbors])

    def _weighted_sum_of_neighbors(self, user, item):
        neighbors = users_adjacency_list[user]
        W_user = self.mean_W[user, neighbors]

        neighbors_R = self._neighbors_R(user, item)
        assert len(W_user) == len(neighbors_R)

        neighbor_has_R = neighbors_R > 0

        wighted_R = np.where(neighbor_has_R, np.multiply(neighbors_R, W_user), 0)
        return wighted_R[neighbor_has_R].sum()

    def _sample_U(self):
        for i in range(n_users):
            items_has_R = np.where(R[i, :] > 0)[0]
            cov = ((self.alpha_U - 1) / self.beta_U) * np.identity(n_latent) + (1 / self.sigma) * (
                sum([self.covs_V[j] + np.dot(np.array([self.mean_V[:, j]]).T, np.array([self.mean_V[:, j]])) for j in
                     items_has_R]))

            self.covs_U[i] = cov

            weighted_R_items = []
            for item in items_has_R:
                weighted_R_items.append(self._weighted_sum_of_neighbors(user=i, item=item))

            mean = np.dot(cov, (
                    (1 / self.sigma) *
                    ((np.dot(R[i, R[i, :] > 0], self.mean_V[:, R[i, :] > 0].T))
                     +
                     (sum(weighted_R_items) * np.sum(self.mean_V[:, R[i, :] > 0], axis=1)))
            ))
            self.mean_U[:, i] = mean

            self.U[:, i] = np.random.multivariate_normal(mean, cov, 1)

    def _sample_V(self):
        for j in range(n_items):
            users_has_R = np.where(R[:, j] > 0)[0]
            cov = ((self.alpha_V - 1) / self.beta_V) * np.identity(n_latent) + (1 / self.sigma) * (
                sum([self.covs_U[i] + np.dot(np.array([self.mean_U[:, i]]).T, np.array([self.mean_U[:, i]])) for i in
                     users_has_R]))

            cov = ((self.alpha_V - 1) / self.beta_V) * np.identity(n_latent) + (1 / self.sigma) * (
                sum([self.covs_U[i] for i in
                     users_has_R]))

            self.covs_V[j] = cov

            weighted_R_users = []
            for user in users_has_R:
                weighted_R_users.append(self._weighted_sum_of_neighbors(user=user, item=j))

            mean = np.dot(cov, (
                    (1 / self.sigma) *
                    ((np.dot(R[R[:, j] > 0, j], self.mean_U[:, R[:, j] > 0].T))
                     +
                     (sum(weighted_R_users) * np.sum(self.mean_U[:, R[:, j] > 0], axis=1)))
            ))

            self.mean_V[:, j] = mean
            self.V[:, j] = np.random.multivariate_normal(mean, cov, 1)

    def _sample_sigma_U(self):
        self.alpha_U += (n_users * n_latent) / 2

        for i in range(n_users):
            for d in range(n_latent):
                self.beta_U += 0.5 * (self.mean_U[d, i] ** 2 + self.covs_U[i][d, d])

        self.sigma_U = 1 / gamma.rvs(self.alpha_U, self.beta_U)
        self.sigma_U = 1

    def _sample_sigma_V(self):
        self.alpha_V += (n_items * n_latent) / 2

        for j in range(n_items):
            for d in range(n_latent):
                self.beta_V += 0.5 * (self.mean_V[d, j] ** 2 + self.covs_U[j][d, d])

        self.sigma_V = 1 / gamma.rvs(self.alpha_V, self.beta_V)
        self.sigma_V = 1

    def _sample_W(self):
        for user in range(n_users):
            neighbors = users_adjacency_list[user]
            for neighbor in neighbors:
                items_has_R = np.where(R[user, :] > 0)[0]

                R_kj = sum([R[neighbor, item] ** 2 for item in items_has_R])
                if R_kj != 0:
                    sigma_inv = ((1 / self.sigma) * R_kj)
                    sigma = 1 / sigma_inv

                    mean = sigma * (
                        sum([R[user, item] * R[neighbor, item] - np.dot(self.mean_U[:, user].T, self.mean_V[:, item]) *
                             R[
                                 neighbor, item] for
                             item in items_has_R]))

                    self.W[user, neighbor] = np.random.normal(mean, sigma)
                else:
                    self.W[user, neighbor] = 0

    def sample(self):
        self._sample_U()
        self._sample_V()
        self._sample_sigma_U()
        self._sample_sigma_V()
        self._sample_W()

        self.mean_U = np.random.normal(0.0, self.sigma_U, (n_latent, n_users))
        self.covs_U = [self.sigma_U * np.identity(n_latent) for _ in range(n_users)]

        self.mean_V = np.random.normal(0.0, self.sigma_V, (n_latent, n_items))
        self.covs_V = [self.sigma_V * np.identity(n_latent) for _ in range(n_items)]

        # self.W = np.random.random(size=(n_users, n_users))
        self.mean_W = np.zeros((n_users, n_users))
        self.sigmas_W = np.zeros((n_users, n_users))

        Ut_V: np.ndarray = np.matmul(self.U.T, self.V)
        r_hat = np.random.normal(Ut_V, self.sigma, size=Ut_V.shape)
        return r_hat


def normalize(a):
    return MinMaxScaler().fit_transform(a) * 5


def compute_training_loss(a, b):
    training = a > 0
    squared_error = np.power(np.where(training, a - b, 0), 2)
    return squared_error[training].mean()


variational_sampler = VariationalMeanField(sigma=np.var(R[R > 0]))

training_loss = []

Ut_V: np.ndarray = np.matmul(variational_sampler.U.T, variational_sampler.V)
r_hat = np.random.normal(Ut_V, variational_sampler.sigma, size=Ut_V.shape)
normalized_r_hat = normalize(r_hat)
loss = compute_training_loss(R, normalized_r_hat)
print(-1, loss)
training_loss.append(loss)
for k in range(n_epochs):
    r_hat = variational_sampler.sample()
    normalized_r_hat = normalize(r_hat)
    loss = compute_training_loss(R, normalized_r_hat)
    training_loss.append(loss)
    print(k, loss)
    print('#' * 20)


def mse(y_pred, y_true):
    return np.power(np.subtract(y_true, y_pred), 2).mean()


def predict(user_id, item_id):
    return normalized_r_hat[user_id, item_id]


ground_truths = []
predictions = []
false_data_count = 0
for _, row in validation_data.iterrows():
    try:
        predictions.append(predict(row.loc['user'], row.loc['item']))
        ground_truths.append(row.loc['score'])
    except IndexError as e:
        false_data_count += 1
        continue

print('mse: ', mse(ground_truths, predictions))
print('false data count: ', false_data_count)

plt.plot(training_loss)
plt.title('training loss')
plt.xlabel('iteration')
plt.ylabel('RMSE')
plt.show()
