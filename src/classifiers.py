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
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))  #softmax numerator = e^(logit[j] + log(K)) where log(K) = -max(logits)
    N = logits.shape[0]     # N = number of samples
    probs /= np.sum(probs, axis=1, keepdims=True)  #softmax denom = sum(e^(logit[j] + log(K))) where we sum for each example (row)
    index = np.arange(N),y  # From every row, choose y[i] column
    li = -np.log(probs[index].clip(min=1e-7))
    loss = np.sum(li) / N
    dlogits = probs.copy()
    dlogits[index] -= 1     #probability error for each corresponding y[i] using one hot encoding (logits[i,y[i]]  1)
    dlogits /= N            #normalise gradient

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
