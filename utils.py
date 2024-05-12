#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:41:15 2024

@author: isdabo
"""

import numpy as np
import matplotlib.pyplot as plt


def degre_col(M):
  """
  Compute the degree of the transpose of a matrix and return the result.
  
  Args:
      M (array): Matrix.
  
  Returns: 
      p*p array: The diagonal matrix whose i-th entry is the sum of the entries of the i-th column of M.
  """
  n = M.shape[0]
  p = M.shape[1]
  D = np.zeros((p,p))
  for i in range(p):
    D[i,i] = sum(M[:,i])
  return D/n

def T(x, Gamma, p, epsilon = 0.0000000001, iter_max = 500000):
  """
  Computes the p*p diagonal matrix T(x) associated to the variance profile Gamma using a fixed-point algorithm.

  Args:
      x (complex/float): The value for which we want to compute T(x).
      Gamma (array): Variance profile associated to T(x).
      p (int): Number of columns of Gamma.
      epsilon (float): Accuracy of the fixed-point algorithm.
      iter_max (int): Maximal number of iterations of the algorithm.

      Returns:
      p*p array: The matrix T(x).
  """
  n = Gamma.shape[0]
  t = np.zeros((p,1))
  t1 = np.ones((p,1))
  iter = 0
  while np.linalg.norm(t-t1) > epsilon and iter < iter_max:
    t1 = t
    t = 1/((Gamma.T/n) @ (1/(1+(Gamma/n)@t))-x)
    iter += 1
  if iter == iter_max:
    print(np.linalg.norm(t-t1))
  return t

def T_tilde(x, Gamma, n, epsilon = 0.0000000001, iter_max = 500000):
  """
  Computes the n*n diagonal matrix T^˜(x) associated to the variance profile Gamma using a fixed-point algorithm.
    
  Args:
      x (complex/float): The value for which we want to compute T^˜(x).
      Gamma (array): Variance profile associated to T^˜(x).
      n (int): Number of lines of Gamma.
      epsilon (float): Accuracy of the fixed-point algorithm.
      iter_max (int): Maximal number of iterations of the algorithm.
        
      Returns:
      n*n array: The matrix T^˜(x).
  """
  t = np.zeros((n,1))
  t1 = np.ones((n,1))
  iter = 0
  while np.linalg.norm(t-t1) > epsilon and iter < iter_max:
    t1 = t
    t = 1/((Gamma/n) @ (1/(1+(Gamma.T/n)@t))-x)
    iter += 1
  if iter == iter_max:
    print(np.linalg.norm(t-t1))
  return t

def kappa(x, Gamma, n, p, epsilon = 0.0000000001):
  """
  Computes the p*p diagonal matrix kappa(x) associated to the variance profile Gamma using the function T_tilde.
    
  Args:
      x (complex/float): The value for which we want to compute T^˜(x).
      Gamma (array): Variance profile associated to T^˜(x).
      n (int): Number of lines of Gamma.
      p (int): Number of columns of Gamma.
        epsilon (float): Accuracy needed to compute T^˜(x).
        
      Returns:
      p*p array: The matrix kappa(x).
  """
  t_tilde = T_tilde(-x, Gamma, n, epsilon = 0.0000000001)
  K = np.zeros((p,1))
  for i in range(p):
    K[i] = np.trace(np.diag(Gamma[:,i]))/np.trace(np.diag(Gamma[:,i])*t_tilde)
  return np.diag(K.T[0])

def T_deriv(x, Gamma, p, epsilon = 0.000000001):
  """
  Computes the derivative of T(x) associated to the variance profile Gamma.
    
  Args:
      x (complex/float): The value for which we want to compute the derivative of T(x).
      Gamma (array): Variance profile associated to T(x).
      p (int): Number of columns of Gamma.
      epsilon (float): Accuracy needed to compute T(x).
        
      Returns:
      p*p array: The derivative of T(x).
  """
  return (T(x - 2*epsilon, Gamma, p) - 8*T(x - epsilon, Gamma, p) + 8*T(x + epsilon, Gamma, p) - T(x + 2*epsilon, Gamma, p))/(12*epsilon)

def T_tilde_deriv(x, Gamma, n, epsilon = 0.000000001):
  """
  Computes the derivative of T^˜(x) associated to the variance profile Gamma.
    
  Args:
      x (complex/float): The value for which we want to compute the derivative of T(x).
      Gamma (array): Variance profile associated to T^˜(x).
      n (int): Number of lines of Gamma.
      epsilon (float): Accuracy needed to compute T^˜(x).
        
      Returns:
      n*n array: The derivative of T^˜(x).
  """
  return (T_tilde(x - 2*epsilon, Gamma, n) - 8*T_tilde(x - epsilon, Gamma, n) + 8*T_tilde(x + epsilon, Gamma, n) - T_tilde(x + 2*epsilon, Gamma, n))/(12*epsilon)

def kappa_deriv(x, Gamma, n, p, epsilon = 0.000000001):
  """
  Computes the derivative of kappa(x) associated to the variance profile Gamma.
    
  Args:
      x (complex/float): The value for which we want to compute the derivative of T(x).
      Gamma (array): Variance profile associated to kappa(x).
      n (int): Number of lines of Gamma.
      p (int): Number of columns of Gamma.
      epsilon (float): Accuracy needed to compute T^˜(x).
        
      Returns:
      p*p array: The derivative of kappa(x).
  """
  return (kappa(x - 2*epsilon, Gamma, n, p) - 8*kappa(x - epsilon, Gamma, n, p) + 8*kappa(x + epsilon, Gamma, n, p) - kappa(x + 2*epsilon, Gamma, n, p))/(12*epsilon)


def variance_profile(name, n, p, tau = 0.05, tau2 = 10, tau3 = 0.1, const = 1, prob = 10, K = 500, param_exp = 1, param_poi = 1, row1 = 1, col1 = 0.005, row2 = 1, col2 = 1, const_band = 1, threshold_diag = 10):
  """
  Computes a variance profile given is name amongst some examples.
    
  Args:
      name (string): The name of the wanted variance profile.
      n (int): Number of lines of Gamma.
      p (int): Number of columns of Gamma.
      tau (float): Parameter for the Piecewise constant \ Block \ Polynomial variance profiles.
      tau2 (float): Parameter for the Block variance profile.
      tau3 (float): Parameter for the Block variance profile.
      const (float): Parameter for the Constant variance profile.
      prob (float): Parameter for the Bernoulli variance profile, it must be greater than 0 and less than 1.
      K (float): Parameter for the Bi-stochastic variance profile.
      param_exp (float): Parameter for the Exponential variance profile.
      param_poi (float): Parameter for the Poisson variance profile.
      row1 (float): Parameter for the Alternated rows variance profile.
      row2 (float): Parameter for the Alternated rows profile.
      col1 (float): Parameter for the Alternated columns profile.
      col2 (float): Parameter for the Alternated columns profile.
      const_band (float): Parameter for the Band profile.
      threshold_diag (int): Parameter for the Band variance profile, it must be positive and less than max(n,p).
        
      Returns:
      n*p array: The wanted variance profile.
  """
  Gamma = np.ones((n,p))
  if name == 'constant':
    Gamma = const * np.ones((n,p))
  if name == 'const_piecewise':
    Gamma = np.ones((n,p))
    Gamma[0:n//4,0:p//4] *= tau
    Gamma[n//4:,p//4:] *= tau
  if name == 'block':
    Gamma = np.ones((n,p))
    Gamma[0:n//4,0:p//4] *= tau
    Gamma[n//4:(n//4+n//3),p//4:(p//4+p//3)] *= tau
    Gamma[n//4+n//3:,(p//4+p//3):] *= tau
    Gamma[0:n//4,p//4:(p//4+p//3)] *= tau2
    Gamma[n//4:n//4+n//3,0:p//4] *= tau2
    Gamma[n//4:(n//4+n//3):,(p//4+p//3):] *= tau3
    Gamma[(n//4+n//3):,p//4:(p//4+p//3)] *= tau3
  if name == 'alternate_row':
    Gamma = np.ones((n,p))
    Gamma[0::2] = row1
    Gamma[1::2] = row2
  if name == 'alternate_column':
    Gamma = np.ones((n,p))
    Gamma[:,0::2] = col1
    Gamma[:,1::2] = col2
  if name == 'bernoulli':
    Gamma =  np.random.binomial(1,prob / p ,size=(n,p))
  if name == 'bernoulli_piecewise':
    Gamma = np.ones((n,p))
    Gamma[0:n//4,0:p//4] = np.random.binomial(1,prob / p ,size=Gamma[0:n//4,0:p//4].shape)
    Gamma[n//4:,p//4:] = np.random.binomial(1,prob / p ,size=Gamma[n//4:,p//4:].shape)
  if name == 'doubly_stochastic':
    Gamma = np.zeros((n,p))
    Inter = np.zeros((n,p))
    if n >= p:
      Inter[0:p, :] = np.eye(p)
      Inter[p:,:] = 1/p
    else:
      Inter[:, 0:n] = np.eye(n)
      Inter[:,n:] = 1/p
    for _ in range(K):
      np.random.shuffle(Inter)
      np.random.shuffle(Inter.T)
      Gamma += p*Inter/K
  if name == 'exponential':
    Gamma = np.random.exponential(param_exp, size = (n,p))
  if name == 'poisson':
    Gamma = np.random.poisson(param_poi, size = (n,p))
  if name == 'polynomial':
    Gamma = np.ones((n,p))
    for i in range(n):
        for j in range(p):
            Gamma[i,j] *= tau+ (np.abs(i-j)/min(n,p))**6
  if name == 'band':
    Gamma = tau * np.ones((n,p))
    for i in range(n):
        for j in range(p):
            if np.abs(i-j) < threshold_diag:
              Gamma[i,j] = const_band
  return (Gamma/(np.sum(np.abs(Gamma))))*p*n


def MSE(n, p, X, lam, profile, alpha = 1,sig = 1):
  """
  Computes the square loss of the ridge estimator on a test sample.
    
  Args:
      n (int): Number of lines of Gamma.
      p (int): Number of columns of Gamma.
      X (n*p array): Training set.
      lam (float): Regularisation parameter of the ridge regression. It must be non-negative.
      profile (string): Name of the Variance profile of the test set.
      alpha (float): Variance of the ridge estimator. It must be positive.
      sig (float): Variance of the error term epsilon. It must be positive.
        
      Returns:
      float: The square loss of the ridge estimator on a test sample.
  """
  r_mse = 0
  for _ in range(200):
    theta = (alpha/np.sqrt(p))* np.random.normal(0,1,size = (p,1))
    Y = X @ theta + np.random.normal(0,sig,size = (n,1))
    if p < n or lam!=0:
      theta_ridge = (np.linalg.inv((X.T) @ X + n*lam*np.eye(p)) @ X.T) @ Y
    elif p>n:
      theta_ridge = ((X.T) @ np.linalg.inv(X@(X.T) + n*lam*np.eye(n))) @ Y
    else:
      print("The ridge estimator cannot be computed.")
    for _ in range(100):
      Gamma_tilde = variance_profile(profile,1,p)
      x = np.sqrt(Gamma_tilde) * np.random.normal(0,1,size = (1,p))
      y = x @ theta + np.random.normal(0,sig)
      r_mse += ((x @ theta_ridge - y)**2)[0][0]
  return r_mse/20000

def R_ts(lam, Gamma, Gamma_test, n, p, alpha = 1, sig = 1, mse = False, profile_test = None, eps = 0.000001):
  """
  Computes the predictive risk, its deterministice equivalent and the square loss associated to the ridge estimator
    
  Args:
      lam (float): Regularisation parameter of the ridge regression. It must be non-negative.
      Gamma (n*p array): Variance profile of the train set.
      Gamma_test (1*p array): Variance profile of the test set.
      n (int): Sample size of the train set.
      p (int): Number of features.
      alpha (float): Variance of the ridge estimator. It must be positive.
      sig (float): Variance of the error term epsilon. It must be positive.
      mse (bool): If True the square loss will be computed, otherwise it will not be computed.
      profile_test (string): Name of the variance profile of the test set.
      eps (float): Accuracy needed to compute T(-lam) and T^˜(-lam).
        
      Returns:
      (float, float, float): (The predictive risk, The deterministic equivalent, The square loss).
  """
  #print("---------- n = "+str(n)+" - p = "+str(p)+" - lambda = "+str(lam)+" ----------")
  X = np.sqrt(Gamma) * np.random.normal(0,1,size = (n,p))
  S = (X.T)@X/n
  Delta_n = degre_col(Gamma)
  Delta_test = degre_col(Gamma_test)
  R = None
  R_mse = None
  if p <= n or lam != 0:
    test = T(-lam, Gamma, p)
    test_pot_min = T(-sig*sig*p/(alpha*alpha*n), Gamma, p)
    test_deriv = T_deriv(-lam, Gamma, p)
    R_T = sig*sig + sig*sig/n * np.trace(Delta_test * test) + lam * ((lam * alpha * alpha / p) - sig*sig/n) * np.trace(Delta_test * test_deriv)
    Q = np.linalg.inv(S+lam*np.eye(p))
    Q2 = Q@Q
    R = sig*sig + (lam*lam*alpha*alpha)/p * np.trace(Delta_test @ Q2) + sig*sig/n * np.trace(Delta_test @ (S @ Q2))
  else:
    K = kappa(lam, Gamma, n, p)
    K_prime = kappa_deriv(lam, Gamma, n, p)
    S_tilde = X@(X.T)/n
    Q_tilde = np.linalg.inv(S_tilde + lam*np.eye(n))
    R_T = sig*sig + alpha*alpha/p * np.trace((K*Delta_test)*  (np.linalg.inv(K+Delta_n))) + ( sig*sig/n - (lam * alpha * alpha / p)) * np.trace(Delta_test * Delta_n * K_prime * np.linalg.inv((K+Delta_n)*(K+Delta_n)))
    R = sig*sig + np.trace(Delta_test @ ((alpha*alpha/p)* np.linalg.matrix_power((X.T/n) @ (Q_tilde @ X) - np.eye(p), 2) + (sig*sig/n) * (X.T/n)@(Q_tilde@(Q_tilde @ X))))
  if mse == True:
    R_mse = MSE(n, p, X, lam+eps, profile_test, alpha,sig)
  return R,R_T, R_mse
