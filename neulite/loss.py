import numpy as np

class LossFunction(object):
    def __init__(self):
        pass

    def loss(self, final_output, labels):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(LossFunction):
    def __init__(self):
        pass

    def loss(self, final_output, labels, epsilon=1e-8):
        num_examples = final_output.shape[0]

        # shift output values for numerical stability
        shifted_output = final_output - np.max(final_output, axis=1, keepdims=True)

        exp_outputs = np.exp(shifted_output)
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

        # numerical stability, clip 0 probabilities
        safe_probs = np.clip(probabilities, epsilon, 1-epsilon)

        cross_entropy = -labels * np.log(safe_probs)

        d_prob = np.copy(probabilities)
        d_prob -= labels
        d_prob /= num_examples
        return probabilities, cross_entropy, d_prob
