import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dlogits. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    normalisation!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    dlogits = np.exp(logits - np.max(logits))  #softmax numerator = e^(logit[j] + log(K)) where log(K) = -max(logits)
    dlogits /= np.sum(dlogits, axis=1, keepdims=True)  #softmax denom = sum(e^(logit[j] + log(K))) where we sum for each example (row)
    N = logits.shape[0]     # N = number of samples
    index = np.arange(N),y
    li = -np.log(dlogits[index])
    loss = np.sum(li) / N
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
