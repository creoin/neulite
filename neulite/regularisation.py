import numpy as np

class Regulariser(object):
    def __init__(self):
        pass

class L2Regulariser(Regulariser):
    def __init__(self, reg_lambda):
        self.reg_lambda = reg_lambda

    def lambda_parameter(self):
        return self.reg_lambda

    def regularisation_loss(self, weight_list):
        reg_loss = 0
        for weights in weight_list:
            reg_loss += 0.5 * self.reg_lambda * np.sum(weights*weights)
        return reg_loss
