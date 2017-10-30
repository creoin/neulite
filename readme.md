# Neulite

Neulite is a light framework for Neural Networks written in Python and
Numpy to shine some light on how they work.

(For the state of the repository on 27/10/2017, see the `initial commit` branch)

## How to run

An example has been included in `run.py`, and is explained briefly below.
From the root directory of the repsitory:

```
import neulite as nl
```

### Setup the data

Firstly setup the data, the Iris dataset has been included in `data/iris`:

```
filepath = 'data/iris/iris.data'
data_manager = nl.DataManager((0.7,0.15,0.15))
data_manager.init_iris(filepath)
X, Y = data_manager.prepare_train()
X_valid, Y_valid = data_manager.prepare_valid()
```

This will read the Iris data, prepare it into a Train/Validation/Test split, by the `(train, valid, test)` tuple supplied to the `DataManager`. With an even random selection over the three classes, and converting the labels to a 1-hot encoding.

### Build the network

Next setup a `NeuralNet` like so:
```
my_net = nl.NeuralNet(4, lr=1e-1)
my_net.set_regulariser(nl.L2Regulariser(1e-3))
my_net.set_cost(nl.SoftmaxCrossEntropyLoss())
```
where the arguments to `NeuralNet` are the input example dimension, and learning rate. Here an `L2Regulariser` with a lambda parameter of `1e-3`, and a `SoftmaxCrossEntropyLoss` has been set.

Next build up the architecture of the layers:
```
my_net.add_layer(nl.FCLayer(100))
my_net.add_layer(nl.ReluLayer(100))
my_net.add_layer(nl.FCLayer(3))
```

### Training

```
num_examples = len(X)
for i in range(10001):
    my_net.feed(X, Y)
    probabilties, loss = my_net.forward()
    d_loss_input = my_net.backward()
    if i % 1000 == 0:
        avg_loss = np.sum(loss)/num_examples
        predicted_class = np.argmax(probabilties, axis=1)
        ground_truth = np.argmax(Y, axis=1)
        train_accuracy = np.mean(predicted_class == ground_truth)
        # similar can be done for X_valid and Y_valid

        print('Iteration {:10}: loss {:6.3f} train accuracy {:7.3f}'.format(i, avg_loss, train_accuracy))
```
