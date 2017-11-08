import pytest
import numpy as np
from numpy.testing import assert_allclose
from neulite.layers import FCLayer

array_tolerance = 1e-3

# Simple example inputs
@pytest.fixture(scope="module")
def ex_inputs():
    inputs = np.array([[1., 2.]])
    return inputs

# Prepare a fixed FC Layer, non-randomised weights
@pytest.fixture(scope="module")
def fc_layer():
    dim_in = 2
    dim_h = 3
    lr = 1e-3
    reg_lambda = 1e-3
    fc_layer = FCLayer(dim_h)
    fc_layer.init(dim_in, lr, reg_lambda)
    fc_layer.biases = np.array([[1.1, 1.2, 1.3]])
    fc_layer.weights = np.array(
        [[1., 2., 3.],
         [4., 5., 6.]]
    )
    return fc_layer


def test_fclayer_feed(ex_inputs, fc_layer):
    fc_layer.feed(ex_inputs)
    assert_allclose(fc_layer.inputs, ex_inputs, rtol=array_tolerance)

# FC Layer forward()
# Check a set of inputs (x) and expected outputs (y) from fixed weights
x1 = np.array([[1., 2.]])
y1 = np.array([[10.1, 13.2, 16.3]])
x2 = np.array([[0.5, 1.]])
y2 = np.array([[5.6, 7.2, 8.8]])
x3 = np.array([[1., 2.], [0.5, 1.]]) # batching above two together
y3 = np.array([[10.1, 13.2, 16.3], [5.6, 7.2, 8.8]]) # batched results
@pytest.mark.parametrize("inputs, expected_output", [
    (x1, y1),
    (x2, y2),
    (x3, y3)
])
def test_fclayer_forward(inputs, expected_output, fc_layer):
    fc_layer.feed(inputs)
    result = fc_layer.forward()
    assert_allclose(result, expected_output, rtol=array_tolerance)

# FC Layer backward() d_inputs, layer weights set in fc_layer fixture
x1   = np.array([[1., 2.]])          # inputs
dl1  = np.array([[0.5, 1.0, 1.5]])   # d_loss/d_y (where y is layer output)
din1 = np.array([[7.0, 16.0]])       # d_loss/d_input
@pytest.mark.parametrize("inputs, dloss, expected_d_input", [
    (x1, dl1, din1)
])
def test_fclayer_backward_d_inputs(inputs, dloss, expected_d_input, fc_layer):
    fc_layer.feed(inputs)
    d_input = fc_layer.backward(dloss)
    assert_allclose(d_input, expected_d_input, rtol=array_tolerance)

# FC Layer backward() d_weights, INCLUDING regularisation - set reg_lambda to 1.0
# Also test weights update
# Seems accurate to the 0.1% level...
x1  = np.array([[1., 2.]])          # inputs
dl1 = np.array([[0.5, 1.0, 1.5]])   # d_loss/d_y (where y is layer output)
dw1 = np.array([[1.5, 3.0, 4.5], [5.0, 7.0, 9.0]])       # d_loss/d_input (+W from regularisation)
@pytest.mark.parametrize("inputs, dloss, expected_d_weights", [
    (x1, dl1, dw1)
])
def test_fclayer_backward_d_weights(inputs, dloss, expected_d_weights, fc_layer):
    # d_weights
    weights_before = np.copy(fc_layer.weights)
    fc_layer.reg_lambda = 1.0
    fc_layer.feed(inputs)
    d_input = fc_layer.backward(dloss)
    d_weights = fc_layer.d_weights
    assert_allclose(d_weights, expected_d_weights, rtol=array_tolerance)

# FC Layer backward() weights update, generate d_weights first as in previous test
@pytest.mark.parametrize("inputs, dloss", [
    (x1, dl1)
])
def test_fclayer_backward_weights_update(inputs, dloss, fc_layer):
    # d_weights
    weights_before = np.copy(fc_layer.weights)
    fc_layer.reg_lambda = 1.0
    fc_layer.feed(inputs)
    d_input = fc_layer.backward(dloss)
    d_weights = fc_layer.d_weights

    # weights update
    lr = fc_layer.lr
    weights_after = fc_layer.weights
    expected_weights = weights_before - (lr * d_weights)
    assert_allclose(weights_after, expected_weights, rtol=array_tolerance)

# FC Layer backward() d_biases
# In current implementation the normalised d_losses are summed
dl3 = np.array([[0.5, 1.0, 1.5], [3.0, 2.0, 1.0]])
db3 = np.array([[3.5, 3.0, 2.5]])
@pytest.mark.parametrize("inputs, dloss, expected_d_biases", [
    (x3, dl3, db3)
])
def test_fclayer_backward_d_biases(inputs, dloss, expected_d_biases, fc_layer):
    fc_layer.feed(inputs)
    d_input = fc_layer.backward(dloss)
    d_biases = fc_layer.d_biases
    assert_allclose(d_biases, expected_d_biases, rtol=array_tolerance)

# FC Layer backward() biases update
@pytest.mark.parametrize("inputs, dloss, expected_d_biases", [
    (x3, dl3, db3)
])
def test_fclayer_backward_biases_update(inputs, dloss, expected_d_biases, fc_layer):
    biases_before = np.copy(fc_layer.biases)
    fc_layer.feed(inputs)
    d_input = fc_layer.backward(dloss)
    d_biases = fc_layer.d_biases

    # biases update
    lr = fc_layer.lr
    biases_after = fc_layer.biases
    expected_biases = biases_before - (lr * d_biases)
    assert_allclose(biases_after, expected_biases, rtol=array_tolerance)
