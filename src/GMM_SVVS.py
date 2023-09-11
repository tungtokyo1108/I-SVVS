#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tungdang
"""

import warnings
from abc import ABCMeta, abstractmethod
from time import time 

import math
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma, gammaln, logsumexp
from scipy import linalg

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn import cluster
from sklearn.utils.extmath import row_norms

#------------------------------------------------------------------------------
# Help functions 
#------------------------------------------------------------------------------

def sigmoid(x):
    "Numerically stable sigmoid function."
    n_samples, n_features = x.shape
    z = np.empty((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            if x[i][j] >= 0:
                z[i][j] = np.exp(-x[i][j])
                z[i][j] = 1 / (1 + z[i][j])
            else:
                z[i][j] = np.exp(x[i][j])
                z[i][j] = z[i][j] / (1 + z[i][j])
    return z

def _estimate_gaussian_covariances_full(resp, X, select, nk, means, reg_covar):
    
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * (select * diff).T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances_tied(resp, X, select, nk, means, reg_covar):
    
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot((nk * means).T, means)
    covariances = avg_X2 - avg_means2
    covariances /= nk.sum(axis = 0)
    covariances.flat[::len(covariances) + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances_diag(resp, X, select, nk, means, reg_covar):
    
    avg_X2 = np.dot(resp.T, select * X * X) / nk
    avg_means2 = (np.dot(resp.T, select) * means ** 2) / nk
    avg_X_means = means * np.dot(resp.T, select * X) / nk
    
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

def _compute_precision_cholesky(covariances, covariance_type):
    
    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape 
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance, lower=True)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T 
    elif covariance_type == 'tied':
        _, n_features = covariances.shape 
        cov_chol = linalg.cholesky(covariances, lower=True)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T 
    else: 
        precisions_chol = 1. / np.sqrt(covariances)
        #precisions_chol = np.nan_to_num(precisions_chol) + 1e-6
        
    return precisions_chol

def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape 
        log_det_chol = (np.log(
            matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]))
    elif covariance_type == "tied":
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))
    else:
        log_det_chol = np.log(matrix_chol)
    
    return log_det_chol

#------------------------------------------------------------------------------
# Stochastic Variational Inferenece for GMM 
#------------------------------------------------------------------------------

class GMM():
    
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3, 
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 weights_init=None, means_init=None, precisions_init=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=42, warm_start=False, verbose=0,
                 verbose_interval=10):
        
        self.n_components = n_components
        self.tol = tol 
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        
        self.covariance_type = covariance_type
        self.weights_init = weights_init 
        self.means_init = means_init 
        self.precisions_init = precisions_init 
        
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior 
        self.mean_precision_prior = mean_precision_prior 
        self.mean_prior = mean_prior 
        self.degrees_of_freedom_prior = degrees_of_freedom_prior 
        self.covariance_prior = covariance_prior 
        
    def _initialize_parameters(self, X, random_state):
        
        n_samples, n_features = X.shape
        
        self.weight_concentration_prior_ = 1./self.n_components
        self.select_prior = 1
        
        self.mean_precision_prior_ = 1.
        self.mean_prior_ = X.mean(axis=0)
        self.degrees_of_freedom_prior_ = n_features
        self.covariance_prior_ = {
            'full': np.atleast_2d(np.cov(X.T)),
            'tied': np.atleast_2d(np.cov(X.T)),
            'diag': np.var(X, axis=0, ddof=1),
            'spherical': np.var(X, axis=0, ddof=1).mean()
        }[self.covariance_type]
        
        random_state = check_random_state(self.random_state)
        
        self.resp = np.zeros((n_samples, self.n_components))
        
        if self.init_params == 'kmeans':
            #resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            self.resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            self.resp = random_state.rand(n_samples, self.n_components)
            self.resp /= self.resp.sum(axis=1)[:, np.newaxis]
        
        select = np.zeros((n_samples, n_features))
        select_non = np.zeros((n_samples, n_features))
        for d in range(n_features):
            chois = random_state.rand(n_samples, 2)
            select[:,d] = chois[:,0]
            select_non[:,d] = chois[:,1]
        select_norm = select/(select + select_non)
        self.selected = select_norm
        
        # eq 10.51, .52, .53
        #nk = self.resp.sum(axis=0) + 10 * np.finfo(self.resp.dtype).eps
        nk = np.dot(self.resp.T, self.selected) + 10 * np.finfo(self.resp.dtype).eps
        #xk = np.dot(self.resp.T, X) / nk[:, np.newaxis]
        xk = np.dot(self.resp.T, self.selected * X) / nk
        sk = {"full": _estimate_gaussian_covariances_full,
                       "tied": _estimate_gaussian_covariances_tied,
                       "diag": _estimate_gaussian_covariances_diag,
                       }[self.covariance_type](self.resp, X, self.selected, nk, xk, self.reg_covar)
        
        self._estimate_weights(nk)
        self._estimate_selection()
        self._estimate_means(nk, xk)
        self._estimate_wishart(nk, xk, sk)
        self._estimate_means_rj(X)
        self._estimate_wishart_rj(X)
        return nk, xk, sk
        
    
        
    def _estimate_weights(self, nk): 
        
        nk = self.resp.sum(axis=0) + 10 * np.finfo(self.resp.dtype).eps
        
        self.weight_concentration_ = (
            1. + nk,
            (self.weight_concentration_prior_ + 
             np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
    
    def _estimate_selection(self):
        
        self.xi1 = self.select_prior + self.selected.sum(axis=0)
        self.xi2 = self.select_prior + (1 - self.selected).sum(axis=0)
    
    def _estimate_means(self, nk, xk):
        # eq 10.60, .61
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = ((self.mean_precision_prior_ * self.mean_prior_ + 
                        nk * xk) / 
                       self.mean_precision_)
        
    def _estimate_means_rj(self, X):
        
        self.mean_precision_rj_ = self.mean_precision_prior_ + (1 - self.selected).sum(axis = 0)
        self.means_rj_ = ((self.mean_precision_prior_ * self.mean_prior_ + ((1 - self.selected) * X).sum(axis = 0)) / self.mean_precision_rj_)
    
    
    def _estimate_wishart(self, nk, xk, sk):
        # eq 10.62, .63
        _, n_features = xk.shape 
        
        if self.covariance_type == 'full':
            
            self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk 
            
            self.covariances_ = np.empty((self.n_components, n_features,
                                          n_features))
            
            for k in range(self.n_components):
                diff = xk[k] - self.mean_prior_
                self.covariances_[k] = (self.covariance_prior_ + nk[k] * sk[k] + 
                                        nk[k] * self.mean_precision_prior_ / 
                                        self.mean_precision_[k] * np.outer(diff, diff))
                self.covariances_[k] = (self.covariances_[k] / self.degrees_of_freedom_[k]) + 1e-6
            
            #self.covariances_ /= (self.degrees_of_freedom_[:, :, np.newaxis])
            
        elif self.covariance_type == 'tied':
            
            self.degrees_of_freedom_ = (self.degrees_of_freedom_prior_ + nk.sum(axis = 0) / self.n_components)
            
            diff = xk - self.mean_prior_
            self.covariances_ = (
                self.covariance_prior_ + sk * nk.sum(axis = 0) / self.n_components + 
                self.mean_precision_prior_ / self.n_components * np.dot(
                    ((nk / self.mean_precision_) * diff).T, diff))
            
            self.covariances_ /= self.degrees_of_freedom_
            
        elif self.covariance_type == 'diag':
            
            self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk 
            
            diff = xk - self.mean_prior_
            self.covariances_ = (
                self.covariance_prior_ + nk * (
                    sk + (self.mean_precision_prior_ / 
                          self.mean_precision_) * np.square(diff)))
            
            self.covariances_ /= self.degrees_of_freedom_
            
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)
        #self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'diag')
        
    def _estimate_wishart_rj(self, X):
        
        n_samples, n_features = X.shape
        
        if self.covariance_type == 'full':
            
            self.degrees_of_freedom_rj_ = self.degrees_of_freedom_prior_ + self.selected.sum(axis = 0)
            
            diff = X - self.means_rj_
            covariance = np.dot((self.selected * diff).T, diff) / self.selected.sum(axis = 0)
            covariance.flat[::n_features + 1] += self.reg_covar
            
            diff_ = self.means_rj_ - self.mean_prior_
            self.covariances_rj_ = (self.selected.sum(axis = 0) * covariance + 
                                    self.selected.sum(axis = 0) * self.mean_precision_prior_ / 
                                    self.mean_precision_rj_ * np.outer(diff_, diff_))
            self.covariances_rj_ = (self.covariances_rj_ / self.degrees_of_freedom_rj_) + 1e-6
            cov_chol = linalg.cholesky(self.covariances_rj_, lower=True) 
            self.precisions_cholesky_rj_ = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
            #self.precisions_cholesky_rj_ = 1. / np.sqrt(self.covariances_rj_)
            #self.precisions_cholesky_rj_ = np.nan_to_num(self.precisions_cholesky_rj_) + 1e-6
            
        elif self.covariance_type == 'diag':
            
            self.degrees_of_freedom_rj_ = self.degrees_of_freedom_prior_ + (1 - self.selected).sum(axis = 0)
            
            nk = (1 - self.selected).sum(axis = 0)
            avg_X2 = ((1 - self.selected) * X * X).sum(axis = 0) / nk
            avg_means2 = ((1 - self.selected) * self.means_rj_ ** 2).sum(axis = 0) / nk
            avg_X_means = (self.means_rj_ * ((1 - self.selected) * X).sum(axis = 0)) / nk
            sk = avg_X2 - 2 * avg_X_means + avg_means2 + self.reg_covar
            
            diff = self.means_rj_ - self.mean_prior_
            self.covariances_rj_ = (nk * (sk + (self.mean_precision_prior_ / self.mean_precision_rj_) * np.square(diff)))
            self.covariances_rj_ /= self.degrees_of_freedom_rj_
            self.precisions_cholesky_rj_ = 1. / np.sqrt(self.covariances_rj_)
            
    def _estimate_log_weights(self):
        
        digamma_sum = digamma(self.weight_concentration_[0] + 
                              self.weight_concentration_[1])
        digamma_a = digamma(self.weight_concentration_[0])
        digamma_b = digamma(self.weight_concentration_[1])
        
        return (digamma_a - digamma_sum + 
                np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
    
    def _estimate_log_prob(self, X):
        
        n_samples, n_features = X.shape
        n_components, _ = self.means_.shape
        
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_, self.covariance_type, n_features)
        
        if self.covariance_type == 'full':
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(self.means_, self.precisions_cholesky_)):
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                log_prob[:, k] = np.sum(np.square(y) * self.selected * self.degrees_of_freedom_[k], axis=1)
                
        elif self.covariance_type == 'tied':
            log_prob = np.empty((n_samples, n_components))
            for k, mu in enumerate(self.means_):
                y = np.dot(X, self.precisions_cholesky_) - np.dot(mu, self.precisions_cholesky_)
                log_prob[:, k] = np.sum(np.square(y), axis=1)
                
        elif self.covariance_type == 'diag':
            precisions = self.degrees_of_freedom_ * (self.precisions_cholesky_ ** 2)
            log_prob = (np.dot(self.selected ,(self.means_ ** 2 * precisions).T) - 
                        2. * np.dot(self.selected * X, (self.means_ * precisions).T) + 
                        np.dot(self.selected * X ** 2, precisions.T)) 
            
        log_gauss = (-.5 * (n_features * np.log(2 * np.pi) + log_prob)) + .5 * np.dot(self.selected, log_det.T) 
                    #- .5 * n_features * np.dot(self.selected, np.log(self.degrees_of_freedom_).T)
        
        log_lambda = n_features * np.log(2.) + digamma(.5 * (self.degrees_of_freedom_ - np.arange(0, n_features)))
        log_lambda = .5 * (log_lambda - n_features / self.mean_precision_)
        
        return log_gauss + np.dot(self.selected, log_lambda.T)
    
    def _estimate_weighted_log_prob(self, X):
        
        return self._estimate_log_prob(X) + self._estimate_log_weights()
    
    def _estimate_log_prob_resp(self, X):
        
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis = 1)
        with np.errstate(under = 'ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    
    def _estimate_log_prob_selected(self, X):
        
        n_samples, n_features = X.shape
        n_components, _ = self.means_.shape
        
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_, self.covariance_type, n_features)
        
        if self.covariance_type == 'full':
            log_prob_selected = np.empty((n_samples, n_features))
            for k, (mu, prec_chol) in enumerate(zip(self.means_, self.precisions_cholesky_)):
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                log_pro_select = np.square(y) * self.degrees_of_freedom_[k]
                log_pro_select = (log_pro_select.T * (self.resp.T)[k]).T
                log_prob_selected += log_pro_select
        
        elif self.covariance_type == 'diag':
            precisions = self.degrees_of_freedom_ * self.precisions_cholesky_ ** 2
            #precisions = precisions_cholesky_ ** 2
            log_prob_selected = (np.dot(self.resp, ((self.means_ ** 2) * precisions)) - 
                        (2. * X * np.dot(self.resp, (self.means_ * precisions))) + 
                        ((X ** 2) * np.dot(self.resp, precisions)))
            
        log_gauss_selected = ((-.5 * (log_prob_selected)) + .5 * np.dot(self.resp, log_det))     
        
        log_lambda =  digamma(.5 * (self.degrees_of_freedom_ - np.arange(0, n_features)))
        log_lambda = .5 * (log_lambda - n_features / self.mean_precision_)
        
        estimate_log_gauss_selected = log_gauss_selected + np.dot(self.resp, log_lambda) 
        estimate_log_gauss_selected = estimate_log_gauss_selected + (digamma(self.xi1) - digamma(self.xi1 + self.xi2))
        
        return estimate_log_gauss_selected
    
    def _estimate_log_prob_rejected(self, X):
        
        n_samples, n_features = X.shape
        
        if self.covariance_type == 'full':
            log_det = np.log(np.diag(self.precisions_cholesky_rj_))
            
            y = np.dot(X, self.precisions_cholesky_rj_) - np.dot(self.means_rj_, self.precisions_cholesky_rj_)
            log_prob_rejected = np.square(y) * self.degrees_of_freedom_rj_ 
        
        elif self.covariance_type == 'diag':    
            log_det = np.log(self.precisions_cholesky_rj_)
            
            precisions_rj_ = self.degrees_of_freedom_rj_ * self.precisions_cholesky_rj_ ** 2
            log_prob_rejected = (((self.means_rj_ ** 2) * precisions_rj_) - 
                        (2. * X * (self.means_rj_ * precisions_rj_)) + 
                        ((X ** 2) * precisions_rj_)) 
        
        log_gauss_rejected = ((-.5 * (log_prob_rejected)) + .5 * log_det) 

        log_lambda_rejected =  digamma(.5 * (self.degrees_of_freedom_rj_ - np.arange(0, n_features)))
        log_lambda_rejected = .5 * (log_lambda_rejected - n_features / self.mean_precision_rj_)

        estimate_log_gauss_rejected = log_gauss_rejected + log_lambda_rejected
        estimate_log_gauss_rejected = estimate_log_gauss_rejected + (digamma(self.xi2) - digamma(self.xi1 + self.xi2))
        
        return estimate_log_gauss_rejected
    
    def _estimate_prob_selection(self, X):
        
        selection = self._estimate_log_prob_selected(X)
        rejection = self._estimate_log_prob_rejected(X)
        
        select_exp = np.exp(selection)
        select_exp = np.nan_to_num(select_exp, posinf=1)
        
        reject_exp = np.exp(rejection)
        reject_exp = np.nan_to_num(reject_exp, posinf=1)
        
        #select_exp = sigmoid(selection)
        #reject_exp = sigmoid(rejection)
        
        self.selected = (select_exp + 1e-6) / (select_exp + reject_exp + 1e-6)
        
        return self.selected
    
    def _e_step(self, X):
        
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        prob_selected = self._estimate_prob_selection(X)
        return np.mean(log_prob_norm), log_resp, prob_selected
    
    def _m_step(self, X, log_resp):
        
        n_samples, n_features = X.shape
        
        self.resp = np.exp(log_resp)
        #nk = self.resp.sum(axis=0) + 10 * np.finfo(self.resp.dtype).eps
        #xk = np.dot(self.resp.T, X) / nk[:, np.newaxis]
        nk = np.dot(self.resp.T, self.selected) + 10 * np.finfo(self.resp.dtype).eps
        xk = np.dot(self.resp.T, X * self.selected) / nk
        sk = {"full": _estimate_gaussian_covariances_full,
              "tied": _estimate_gaussian_covariances_tied,
              "diag": _estimate_gaussian_covariances_diag,
             }[self.covariance_type](self.resp, X, self.selected, nk, xk, self.reg_covar)
        
        self._estimate_weights(nk)
        self._estimate_selection()
        self._estimate_means(nk, xk)
        self._estimate_wishart(nk, xk, sk)
        self._estimate_means_rj(X)
        self._estimate_wishart_rj(X)
        
    def fit_predict(self, X):
        
        n_init = self.n_init
        max_lower_bound = -np.infty 
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        
        for init in range(n_init):
            
            self._initialize_parameters(X, random_state)
            lower_bound = -np.infty 
            
            for n_iter in range(1, self.max_iter + 1):
                print(n_iter)
                prev_lower_bound = lower_bound 
                log_prob_norm, log_resp, prob_selected = self._e_step(X)
                self._m_step(X, log_resp)
                
        _, log_resp, prob_selected = self._e_step(X)
        
        return log_resp, prob_selected
