from collections import defaultdict
import tensorflow as tf
import numpy as np


class GaussianMixtureSGD(object):

    def __init__(self, n_components=5, random_seed=0):
        self.n_components = n_components
        self.random_seed = random_seed

    def _make_model(self, n_features, dtype=np.float32):
        self._component_variables = defaultdict(list)
        X = tf.placeholder(shape=(None, n_features), dtype=dtype, name='X')

        # Mixture weights
        w = tf.Variable(tf.zeros(shape=(1, self.n_components),
                                 dtype=dtype), name='w')
        self._normalized_weights = tf.reshape(tf.nn.softmax(w),
                                              (self.n_components,))
        logliks = []

        # TODO: instead of masking using a numpy initialized densed tensor, use
        # a sparse tensorflow tensor with the triangular structure built-in bu
        # this would equire tensorflow >= 0.9 which is not released at this
        # point.
        M = tf.constant(np.tril(np.ones(shape=(n_features, n_features), dtype=dtype),
                                k=-1),
                        name='triangular_mask')
        for k in range(self.n_components):
            with tf.variable_scope('component_%03d' % k):
                mu = tf.Variable(tf.zeros(shape=(n_features,), dtype=dtype),
                                 name='mu_%03d' % k)
                self._component_variables['mu'].append(mu)
                d = tf.Variable(tf.truncated_normal(shape=(n_features,),
                                                    stddev=1 / sqrt(n_features),
                                                    dtype=dtype,
                                                    seed=self.random_seed),
                                name='d_%03d' % k)

                self._component_variables['d'].append(d)
                H = tf.Variable(
                    tf.truncated_normal(shape=(n_features, n_features),
                                                    stddev=1 / sqrt(n_features),
                                                    dtype=dtype,
                                                    seed=self.random_seed),
                                name='H_%03d' % k)
                # M is an element-wise mask to set all diagonal and triangular uppper entries of
                # of H to zero:
                L = tf.add(tf.diag(tf.exp(d)), tf.mul(M, H), name='L_%03d' % k)
                P = tf.matmul(L, tf.transpose(L), name='P_%03d' % k)
                self._component_variables['P'].append(P)

                loglik = self._log_likelihood_one_gaussian(X, mu, P, d)
                logliks.append(loglik)

        # compute the log likelihood of the mixture
        self._loglik = tf.reduce_sum(
            tf.mul(self._normalized_weights, tf.pack(logliks)),
            reduction_indices=1
        )
        # TODO: self._train_opt =

    def _log_likelihood_one_gaussian(self, X, mu, P, d):
        X_mu = X - mu
        X_muTPX_mu = tf.reduce_sum(tf.mul(X_mu, tf.matmul(X_mu, P)),
                                   reduction_indices=1)
        # logdet(C) = -logdet(P) as C is the inverse of P
        # logdet(P) = 2 * logdet(L) = 2 * sum_i d_i
        return (-0.5 * n_features * tf.log(2 * np.pi)
                + tf.reduce_sum(d)
                - 0.5 * X_muTPX_mu)

    def fit(X_train):
        n_samples, n_features = X_train.shape
        # TODO
