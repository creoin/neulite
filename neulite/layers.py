import numpy as np

class Layer(object):
    def feed(self, inputs):
        self.inputs = inputs
        return True

    def n_hidden(self):
        return self.n_h

    def layer_weights(self):
        return self.weights

    def output(self):
        return self.outputs

    def init(self, dim_input, lr, reg_lambda):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, dim_h):
        self.biases  = np.random.random((1, dim_h))*0.1
        # self.biases = np.zeros((1,dim_h))
        self.outputs = np.zeros(dim_h)
        self.n_h = dim_h

    def init(self, dim_input, lr, reg_lambda):
        """
        Initialise layer with additional details that the Network will know about
        """
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(dim_input, self.n_h))
        self.inputs  = np.zeros(dim_input)
        self.reg_lambda = reg_lambda
        self.lr = lr
        print('Created layer: {} inputs, {} neurons'.format(dim_input, self.n_h))

    def forward(self):
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, del_loss):
        self.d_inputs = np.dot(del_loss, self.weights.T)
        self.d_weights = np.dot(self.inputs.T, del_loss)
        self.d_weights += self.reg_lambda * self.weights
        self.d_biases = np.sum(del_loss, axis=0, keepdims=True)

        # parameter update
        self.weights += -self.lr * self.d_weights
        self.biases  += -self.lr * self.d_biases
        return self.d_inputs


class ReluLayer(Layer):
    def __init__(self, dim_h):
        self.outputs = None
        self.inputs = None
        self.weights = 0  # to bypass regularisation (e.g. L2)
        self.n_h = dim_h

    def init(self, dim_input, lr, reg_lambda):
        pass

    def forward(self):
        self.outputs = np.maximum(0,self.inputs)
        return self.outputs

    def backward(self, del_loss):
        drelu = np.where(self.inputs > 0, del_loss, 0)
        return drelu


class DropoutLayer(Layer):
    def __init__(self, dim_h, keep_prob=1.0):
        self.outputs = None
        self.inputs = None
        self.weights = 0  # to bypass regularisation (e.g. L2)
        self.n_h = dim_h
        self.keep_prob = keep_prob
        self.keep_mask = None

    def init(self, dim_input, lr, reg_lambda):
        pass

    def forward(self):
        # inverted dropout, to prevent scaling output at test time, 1 / keep_prob
        self.keep_mask = (np.random.uniform(0., 1., self.n_h) < self.keep_prob) / self.keep_prob
        self.outputs = self.inputs * self.keep_mask
        return self.outputs

    def backward(self, del_loss):
        d_drop = del_loss * self.keep_mask
        return d_drop
