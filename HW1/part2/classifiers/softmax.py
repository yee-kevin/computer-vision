import numpy as np
from random import shuffle

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    fx = np.dot(X,W)
    fx_sample_num = fx.shape[0] #500
    fx_class_num = fx.shape[1] #10

    for i in range(fx_sample_num):
      fx_diff = fx[i] - np.max(fx[i])
      fx_sum = np.sum(np.exp(fx_diff))
      for j in range(fx_class_num):
        # If true label
        if j == y[i]:
          # calculate loss
          loss -= np.log(np.exp(fx_diff[j]) / fx_sum)
          dW[:,j] += np.dot((np.exp(fx_diff[j]) / fx_sum - 1),X[i])
        else:
          dW[:,j] += np.dot((np.exp(fx_diff[j]) / fx_sum), X[i])
    dW /= X.shape[0]
    loss /= X.shape[0]
    # using L2 norm as regularisation
    regularisation_loss = 0
    for i in range(W.shape[0]):
      for j in range(W.shape[1]):
        regularisation_loss += np.square(W[i,j])
        dW[i,j] += reg * W[i,j]
    loss += 0.5 * reg * regularisation_loss
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    import copy
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################    
    sample_num = X.shape[0]

    fx = np.dot(X,W)
    fx = np.exp(fx - np.max(fx, axis=1)[:,np.newaxis])

    fx_sum = np.sum(fx, axis=1)
    fx = fx / np.expand_dims(fx_sum,axis=1)

    fx_grad_cal = copy.copy(fx)
    fx_grad_cal[np.arange(fx.shape[0]),y] -= 1

    dW = np.dot(X.T, fx_grad_cal)
    loss = np.sum(-np.log(fx[np.arange(fx.shape[0]),y]))
    
    dW /= sample_num
    loss /= sample_num

    loss += 0.5 * reg * np.sum(W*W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

