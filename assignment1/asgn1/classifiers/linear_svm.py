import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dim = W.shape
  dW = np.zeros(dim) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        transposed_X = X[i].T
        dW[:, y[i]] = dW[:, y[i]] - transposed_X
        dW[:, j] = dW[:, j] + transposed_X
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW = dW + reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  #print(W.shape, X.shape)
  #print(X)
  dim  = W.shape
  dW = np.zeros(dim) # initialize the gradient as zero
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)     # (500 * 3073).(3073 * 10) = 500*10
  #print(scores.shape)
  trained_range = range(num_train)
  correct_class_score = scores[trained_range, y]
  #print(correct_class_score.shape) (N,)
  margin = np.maximum(0, (np.transpose(scores) - correct_class_score + 1))
  margin[y, trained_range] = 0
  #print(margin.shape)
  loss = np.sum(margin)/num_train
  #loss /= num_train
  regularized_term = 0.5*reg*np.sum(W*W)
  loss += regularized_term 
  # now implementing loss
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  #                                                                    #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  loss_grad = np.zeros((scores.shape[0], scores.shape[1]))
  loss_grad[margin.T > 0] = 1
  pos_losses  = -1*np.sum(margin.T > 0, axis = 1)
  #print(pos_losses.shape)
  loss_grad[trained_range,  y] = pos_losses
  transposed_X = X.T
  dW = transposed_X.dot(loss_grad)
  dW = dW/num_train
  dW = dW + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
