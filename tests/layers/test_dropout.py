import pytest
import numpy as np
from neulite.layers import DropoutLayer

@pytest.fixture(scope="module")
def dropout_layer():
    dim_h = 100
    dropout_layer = DropoutLayer(dim_h, keep_prob=0.5)

    dropout_layer.inputs = np.ones((1,100))*10.

    # print(dropout_layer.inputs)
    return dropout_layer

# monkeypatch random element (np.random.uniform)
def mock_random(start, stop, dim):
    # instead of random, send 50 low and 50 high probabilities
    random_array = []
    for i in range(int(dim/2)):
        random_array.append(0.01)
    for i in range(int(dim/2)):
        random_array.append(0.99)
    return np.array(random_array)

def test_dropout_mask(monkeypatch, dropout_layer):
    monkeypatch.setattr(np.random, 'uniform', mock_random)

    dropout_layer.forward()
    mask = dropout_layer.keep_mask
    dim_h = dropout_layer.n_h

    # should be 50% 0, and 50% re-weighted to 2.0
    unique, counts = np.unique(mask, return_counts=True)
    mask_counts = dict(zip(unique, counts))

    assert mask_counts[0.0] == int(dim_h/2)
    assert mask_counts[2.0] == int(dim_h/2)

def test_dropout_output(monkeypatch, dropout_layer):
    monkeypatch.setattr(np.random, 'uniform', mock_random)

    dropout_layer.forward()
    outputs = dropout_layer.outputs
    dim_h = dropout_layer.n_h

    # should be 50% 0, and 50% re-weighted to 20
    unique, counts = np.unique(outputs, return_counts=True)
    output_counts = dict(zip(unique, counts))

    assert output_counts[0.0] == int(dim_h/2)
    assert output_counts[20.0] == int(dim_h/2)
