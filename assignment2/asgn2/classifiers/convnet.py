import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class MyAwesomeNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, num_filters, hidden_dim, input_dim=(3, 32, 32), filter_size=5,
               num_classes=10, weight_scale=1e-3, reg=0.0, dropout=0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dropout_param = {}
    self.dropout_param = {'mode':'train', 'p':dropout}
    self.bn_params = []
    self.bn_params = [{'mode':'train'}, {'mode':'train'}, {'mode':'train'}]
    self.reg = reg
    self.dtype = dtype
    self.num_filters = num_filters
    self.filter_size = filter_size

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    self.params["W1"] = weight_scale * np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size)
    self.params["b1"] = np.zeros(num_filters[0])
    self.params["gamma1"] = np.ones(num_filters[0])
    self.params["beta1"] = np.zeros(num_filters[0])
    self.params["W2"] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size, filter_size)
    self.params["b2"] = np.zeros(num_filters[1])
    #self.params["gamma2"] = np.ones(num_filters[1])
    #self.params["beta2"] = np.zeros(num_filters[1])
    #print(num_filters[1]*(input_dim[1]/4)*(input_dim[2]/4), hidden_dim)
    self.params["W3"] = weight_scale * np.random.randn(num_filters[1]*(input_dim[1]/4)*(input_dim[2]/4), hidden_dim)
    self.params["b3"] = np.zeros(hidden_dim)
    self.params["gamma3"] = np.ones(hidden_dim)
    self.params["beta3"] = np.zeros(hidden_dim)
    self.params["W4"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b4"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma1, beta1 = self.params["gamma1"], self.params["beta1"]
    #gamma2, beta2 = self.params["gamma2"], self.params["beta2"]
    gamma3, beta3 = self.params["gamma3"], self.params["beta3"]

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    if y is None:
        for params in self.bn_params:
            params[mode] = "test"
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.     
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax                                                           #
    ############################################################################
    #pass
    #conv_bactchnorm_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param)
    out, cache_conv1     = conv_bactchnorm_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, self.bn_params[0])
    out, cache_conv2     = conv_relu_pool_forward(out, W2, b2, conv_param, pool_param)
    out, cachce_brdf          = affine_batchnorm_relu_dropout_forward(out, W3, b3, gamma3, beta3, self.bn_params[2], self.dropout_param)
    # out, cache_fc_batch  = batchnorm_forward(out, gamma3, beta3, self.bn_params[2])
    # out, cache_relu      = affine_relu_forward(out, W3, b3)
    # out, cache_dropout   = dropout_forward(out, self.dropout_param)
    scores, cache_affine = affine_forward(out, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #pass
    loss, dScore = softmax_loss(scores, y)
    print(loss)
    # Regularized loss
    loss = loss + 0.5 * self.reg * np.sum(self.params["W1"] ** 2) + 0.5 * self.reg * np.sum(self.params["W2"] ** 2) + 0.5 * self.reg * np.sum(self.params["W3"] ** 2) + 0.5 * self.reg * np.sum(self.params["W4"] ** 2)
    dx, grads["W4"], grads["b4"] = affine_backward(dScore, cache_affine)
    grads["W4"] += self.reg * self.params["W4"]
    dx, grads["W3"], grads["b3"] = conv_relu_pool_backward(dx, cache_conv2)
    grads["W2"] += self.reg * self.params["W2"]
    dx = dropout_backward(dx, cache_dropout)
    dx, grads["W3"], grads["b3"] = affine_relu_backward(dx, cache_relu)
    grads["W3"] += self.reg * self.params["W3"]
    dx, grads['gamma3'], grads['beta3'] = batchnorm_backward(dx, cache_fc_batch)

    dx, grads["W1"], grads["b1"], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dx, cache_conv1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
