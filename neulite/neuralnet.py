import numpy as np

class NeuralNet(object):
    def __init__(self, dim_inputs, lr=1e-1):
        self.n_layers = 0
        self.layers = []
        self.dim_inputs = dim_inputs
        self.final_output = None
        self.lr = lr

    def add_layer(self, layer):
        if self.n_layers == 0:
            n_in = self.dim_inputs
        else:
            n_in = self.layers[self.n_layers - 1].n_hidden()
        layer.init(n_in, self.lr, self.regulariser.lambda_parameter())
        self.layers.append(layer)
        self.n_layers += 1

    def feed(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    # Learning Rate
    def set_lr(self, lr):
        self.lr = lr

    def set_cost(self, cost_function):
        self.cost_function = cost_function

    def set_regulariser(self, regulariser):
        self.regulariser = regulariser

    def forward(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.feed(self.inputs)
            else:
                layer.feed(step_output)
            step_output = layer.forward()
        self.final_output = step_output

        total_loss, prob = self.loss()
        self.prob = prob

        return self.prob, total_loss

    def backward(self):
        d_loss_input = self.del_loss
        for i, layer in reversed(list(enumerate(self.layers))):
            d_loss_input = layer.backward(d_loss_input)
        return d_loss_input

    def loss(self):
        # collect weights
        layer_weights = []
        for layer in self.layers:
            layer_weights.append(layer.layer_weights())
        reg_loss = self.regulariser.regularisation_loss(layer_weights)

        prob, entropy_loss, del_loss = self.cost_function.loss(self.final_output, self.labels)
        self.del_loss = del_loss

        self.total_loss = entropy_loss + reg_loss
        return self.total_loss, prob

    def set_test_mode(self):
        for layer in self.layers:
            layer.set_test_mode()

    def set_train_mode(self):
        for layer in self.layers:
            layer.set_train_mode()
