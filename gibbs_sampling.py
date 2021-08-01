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

n_users = users_items_train_matrix.shape[0]
n_items = users_items_train_matrix.shape[1]

R = users_items_train_matrix.fillna(0).values

n_epochs = 20
n_latent = 2


class GibbsSampler:
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

    def _sample_U(self):
        for i in range(n_users):
            V_j = self.V[:, R[i, :] > 0]
            cov_inv = (1 / self.sigma_U) * np.identity(n_latent) + (1 / self.sigma) + np.dot(V_j, V_j.T)
            cov = np.linalg.inv(cov_inv)
            mean = np.dot(cov, ((1 / self.sigma) * np.dot(R[i, R[i, :] > 0], V_j.T)))

            self.U[:, i] = np.random.multivariate_normal(mean, cov, 1)

    def _sample_V(self):
        for j in range(n_items):
            U_i = self.U[:, R[:, j] > 0]
            cov_inv = (1 / self.sigma_V) * np.identity(n_latent) + (1 / self.sigma) + np.dot(U_i, U_i.T)
            cov = np.linalg.inv(cov_inv)
            mean = np.dot(cov, ((1 / self.sigma) * np.dot(R[R[:, j] > 0, j], U_i.T)))

            self.V[:, j] = np.random.multivariate_normal(mean, cov, 1)

    def _sample_sigma_U(self):
        self.alpha_U += (n_users * n_latent) / 2
        self.beta_U += 0.5 * np.sum(np.power(self.U, 2))
        self.sigma_U = 1 / gamma.rvs(self.alpha_U, self.beta_U)
        self.sigma_U = 1

    def _sample_sigma_V(self):
        self.alpha_V += (n_items * n_latent) / 2
        self.beta_V += 0.5 * np.sum(np.power(self.V, 2))
        self.sigma_V = 1 / gamma.rvs(self.alpha_V, self.beta_V)
        self.sigma_V = 1

    def sample(self):
        self._sample_U()
        self._sample_V()
        self._sample_sigma_U()
        self._sample_sigma_V()
        Ut_V: np.ndarray = np.matmul(self.U.T, self.V)
        r_hat = np.random.normal(Ut_V, self.sigma, size=Ut_V.shape)
        return r_hat


def normalize(a):
    return MinMaxScaler().fit_transform(a) * 5


def compute_training_loss(a, b):
    training = a > 0
    squared_error = np.power(np.where(training, a - b, 0), 2)
    return squared_error[training].mean()


gibbs_sampler = GibbsSampler(sigma=np.var(R[R > 0]))

training_loss = []

Ut_V: np.ndarray = np.matmul(gibbs_sampler.U.T, gibbs_sampler.V)
r_hat = np.random.normal(Ut_V, gibbs_sampler.sigma, size=Ut_V.shape)
normalized_r_hat = normalize(r_hat)
loss = compute_training_loss(R, normalized_r_hat)
training_loss.append(loss)

print(-1, loss)
for k in range(n_epochs):
    r_hat = gibbs_sampler.sample()
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
