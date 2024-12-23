#An Index object acts as an identifier.
import string
import random
import numpy as np

class Index():
    # It has a label and a dimension. If no label is provided, a random string is generated.
    def __init__(self, dim, label=None):
        if label is None:
            self.label = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        else:
            self.label = label
        self.dim = dim
    
    


# A Tensor object contains a jax array and an index collection.
class Tensor():

    def __init__(self, W, inds = None):
        if inds is None:
            self.inds = []
            for i in range(len(W.shape)):
                self.inds.append(Index(W.shape[i]))
        # Constructor will throw an error if index collection given doesn't match shape of jax array
        elif len(W.shape) != len(inds):
            raise Exception('Tensor shape does not match number of indices given')
        else:
            for i in range(len(W.shape)):
                if W.shape[i] != inds[i].dim:
                    raise Exception('Index ' + str(i) + ' does not match array index')
            self.inds = inds
        self.tensor = W

    # takes a jax.ops.index input and a value
    def setElement(self, index, val):
        self.tensor[index] = val

    def printIndices(self):
        for i in range(len(self.inds)):
            print('Label: ' + self.inds[i].label + '  Dimension: ' + str(self.inds[i].dim))
