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
  #print(W.shape[1])
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for index in range(num_train):
    scores = X[index].dot(W)
    scores  = scores - np.max(scores)
    numerator = np.exp(scores)
    denominator = np.sum(np.exp(scores))
    softMax = numerator/denominator
    L_i = -1*np.log(softMax[y[index]])
    loss = loss + L_i
    for j in range(num_classes):
      if j == y[index]:
        dW[:, j] += (softMax[j] - 1)*X[index]
      else:
        dW[:, j] += (softMax[j])*X[index]
  loss /= num_train
  loss = loss + 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg*W

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  """
  W = (D, C)
  X = (N, D)
  y = (N, )
  """
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #scores = np.zero(())
  scores = X.dot(W)
  #print(X.shape, W.shape, scores.shape)
  maxedScores = np.max(scores, axis = 1,  keepdims = True)
  scores =  scores - maxedScores
  #print(scores.shape)
  numerator = np.exp(scores)
  denominator = np.sum(np.exp(scores), axis = 1,  keepdims = True)
  softMax = (numerator*1.0)/(denominator*1.0)
  #print(softMax)
  # print(softMax.shape)
  classes = range(num_train)
  #loss =  np.sum(np.log(scores[classes, y]))
  loss =  -np.sum(np.log(softMax[classes, y]))
  loss /= num_train
  loss = loss + 0.5*reg*np.sum(W*W)
  # Loss Working 
  softMax[classes, y] = softMax[classes, y] - 1
  #dW = (X.T).dot(Scores) # Wrong
  dW = (X.T).dot(softMax) # Softmax but scores
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

