from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
from copy import deepcopy

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
      scores = np.dot(W.T,X[i])
      unnorm_log_prob = np.exp(scores)
      norm_log_prob = unnorm_log_prob/np.sum(unnorm_log_prob)
      loss -= np.log(norm_log_prob[y[i]]) 
      
      for j in range(num_classes):
        if j == y[i]:
          continue
        dW[:,j] += norm_log_prob[j]*X[i] 
      dW[:,y[i]] += (norm_log_prob[y[i]] - 1)*X[i] 
    
    loss/=num_train
    dW /= num_train
    dW += reg*W
    loss+= reg*np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X, W)
    unnorm_log_prob = np.exp(scores)
    norm_log_prob = np.divide(unnorm_log_prob,np.sum(unnorm_log_prob, axis = 1).reshape(num_train,1))
    mask = deepcopy(norm_log_prob)
    mask[np.arange(0,num_train), y] = mask[np.arange(0,num_train), y] - 1
    dW = np.dot(X.T,mask)
    loss = np.sum(-np.log(norm_log_prob[np.arange(0, num_train), y]))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss/=num_train
    dW /= num_train
    dW += reg*W
    loss+= reg*np.sum(W*W)
    return loss, dW
