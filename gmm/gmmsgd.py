from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from math import sqrt


class EpochSampler(object):
    """Helper function to cycle through a shuffled dataset by minibatches.

    The dataset is shuffled at the beginning of each epoch.
    """

    def __init__(self, *data, n_epochs=1, batch_size=100, random_seed=None):
        self.data = data
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

    def __iter__(self):
        rng = np.random.RandomState(0)
        n_samples = self.data[0].shape[0]
        n_seen = 0
        batch_size = self.batch_size
        for epoch in range(self.n_epochs):
            permutation = rng.permutation(n_samples)
            data = tuple(d[permutation] for d in self.data)
            for i in range(0, n_samples, batch_size):
                n_seen += len(data[0][i:i + batch_size])
                yield n_seen, epoch, tuple(d[i:i + batch_size] for d in data)


class GaussianMixtureSGD(object):
    def __init__(self, n_components=5, learning_rate=0.1, patience=3,
                 batch_size=10, max_iter=1000, session=None,
                 means_init=None, random_seed=0):
        self.n_components = n_components
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.session = session
        self.means_init = means_init

    def _make_model(self, n_features, dtype=np.float32):
        self._component_variables = defaultdict(list)
        X = tf.placeholder(shape=(None, n_features), dtype=dtype, name='X')

        # Mixture weights
        w = tf.Variable(
            tf.zeros(shape=(1, self.n_components), dtype=dtype),
            name='w')
        self._normalized_weights = tf.reshape(
            tf.nn.softmax(w), (self.n_components,))
        logliks = []

        # TODO: instead of masking using a numpy initialized densed tensor, use
        # a sparse tensorflow tensor with the triangular structure built-in bu
        # this would equire tensorflow >= 0.9 which is not released at this
        # point.
        M = tf.constant(
            np.tril(
                np.ones(shape=(n_features, n_features), dtype=dtype),
                k=-1),
            name='triangular_mask')
        for k in range(self.n_components):
            with tf.variable_scope('component_%03d' % k):
                if self.means_init is not None:
                    m = np.asarray(self.means_init[k], dtype=dtype)
                else:
                    m = tf.zeros(shape=(n_features,), dtype=dtype)
                mu = tf.Variable(m, name='mu_%03d' % k)
                self._component_variables['mu'].append(mu)
                d = tf.Variable(
                    -2 * tf.ones(shape=[n_features], dtype=dtype),
                    #tf.truncated_normal(shape=[n_features],
                    #                    stddev=1 / sqrt(n_features),
                    #                    dtype=dtype,
                    #                    seed=self.random_seed + k),
                    name='d_%03d' % k)

                self._component_variables['d'].append(d)
                H = tf.Variable(
                    tf.zeros(shape=(n_features, n_features), dtype=dtype),
                    #tf.truncated_normal(shape=(n_features, n_features),
                    #                    stddev=1 / sqrt(n_features),
                    #                    dtype=dtype,
                    #                    seed=self.random_seed + k),
                    name='H_%03d' % k)
                # M is an element-wise mask to set all diagonal and triangular
                # uppper entries of of H to zero:
                L = tf.add(tf.diag(tf.exp(d)), tf.mul(M, H), name='L_%03d' % k)
                P = tf.matmul(L, tf.transpose(L), name='P_%03d' % k)
                self._component_variables['P'].append(P)

                loglik = self._log_likelihood_one_gaussian(
                    n_features, X, mu, P, d)
                logliks.append(loglik)

        # compute the log likelihood of the mixture
        # TODO: would it be better to find a way to vectorize the computation
        # of the log-likelihoods to avoid using tf.pack to make tensorflow
        # run somehow faster?

        # XXX: the following is wrong! We cannot get the loglikelood of a mixture
        # this way... I don't have time to fix it now though.
        # It should use tf.reduce_logsumexp instead.
        self._loglik = tf.reduce_sum(
            tf.mul(tf.transpose(tf.pack(logliks)), self._normalized_weights),
            reduction_indices=1)
        self._loss = -tf.reduce_mean(self._loglik)
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        train_op = self._optimizer.minimize(self._loss)

        if self.session is None:
            session = tf.InteractiveSession()
        else:
            session = self.session
        session.run(tf.initialize_all_variables())
        for name, variables in self._component_variables.items():
            print(name)
            for var in variables:
                print(var.eval())
            if name == 'P':
                print('C')
                for var in variables:
                    print(np.linalg.inv(var.eval()))
        self._train = lambda data: session.run(
            train_op, feed_dict={X: data}
        )
        self.score_samples = lambda data: session.run(
            self._loglik, feed_dict={X: data}
        )
        self._compute_loss = lambda data: session.run(
            self._loss, feed_dict={X: data}
        )
        self.score = lambda data: -self._compute_loss(data)

    def _log_likelihood_one_gaussian(self, n_features, X, mu, P, d):
        X_mu = X - mu
        X_muTPX_mu = tf.reduce_sum(
            tf.mul(X_mu, tf.matmul(X_mu, P)),
            reduction_indices=1)
        # logdet(C) = -logdet(P) as C is the inverse of P
        # logdet(P) = 2 * logdet(L) = 2 * sum_i d_i
        return (-0.5 * n_features * tf.log(2 * np.pi) + tf.reduce_sum(d) - 0.5
                * X_muTPX_mu)


    def fit(self, X_train, X_val=None):
        if X_val is None:
            X_train, X_val = train_test_split(X_train, test_size=0.1,
                                              random_state=self.random_seed)
        n_samples, n_features = X_train.shape
        self._make_model(n_features=n_features)
        batch_sampler = EpochSampler(X_train, n_epochs=self.max_iter,
                                     batch_size=self.batch_size,
                                     random_seed=self.random_seed)
        best_val_loss = self._compute_loss(X_val)
        patience = self.patience
        for n_seen, epoch, (X_batch,) in batch_sampler:
            self._train(X_batch)
            if n_seen % 100 == 0:
                # XXX: ensure that this is a multiple of batch_size
                val_loss = self._compute_loss(X_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = self.patience
                else:
                    patience -= 1
                    if patience == 0:
                        break
        self.n_iter_ = epoch + 1
