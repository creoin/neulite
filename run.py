import numpy as np
import neulite as nl
import os

import csv

# Prepare Dataset
filepath = 'data/iris/iris.data'
data_manager = nl.DataManager((0.7,0.15,0.15))
data_manager.init_iris(filepath)
X, Y = data_manager.prepare_train()
X_valid, Y_valid = data_manager.prepare_valid()

# label for logging files
experiment = 'tmp'
# experiment = 'fc100_fc3_relu'
# experiment = 'fc10_fc3_relu'

# Build Neural Network
my_net = nl.NeuralNet(4, lr=1e-1)
my_net.set_regulariser(nl.L2Regulariser(1e-3))
my_net.set_cost(nl.SoftmaxCrossEntropyLoss())

my_net.add_layer(nl.FCLayer(100))
my_net.add_layer(nl.ReluLayer(100))
my_net.add_layer(nl.DropoutLayer(100, keep_prob=0.7))
my_net.add_layer(nl.FCLayer(3))

# Validation check on neural net
def check_valid(neural_net):
    neural_net.feed(X_valid, Y_valid)
    valid_prob, valid_loss = neural_net.forward()

    predicted_class = np.argmax(valid_prob, axis=1)
    ground_truth = np.argmax(Y_valid, axis=1)
    valid_accuracy = np.mean(predicted_class == ground_truth)
    return valid_accuracy

# Set up a logger for the training data we want to record
quantities = ['Iteration', 'Train_Accuracy', 'Valid_Accuracy', 'Loss']
train_logs = nl.Logger(*quantities)

num_examples = len(X)
print('num_examples: {}'.format(num_examples))
for i in range(10001):
    my_net.feed(X, Y)
    probabilties, loss = my_net.forward()
    d_loss_input = my_net.backward()
    if i % 100 == 0:
        avg_loss = np.sum(loss)/num_examples
        predicted_class = np.argmax(probabilties, axis=1)
        ground_truth = np.argmax(Y, axis=1)
        train_accuracy = np.mean(predicted_class == ground_truth)

        valid_accuracy = check_valid(my_net)

        train_logs.log(Iteration=i, Train_Accuracy=train_accuracy, Valid_Accuracy=valid_accuracy, Loss=avg_loss)

        print('Iteration {:10}: loss {:6.3f} train accuracy {:7.3f} valid accuracy {:7.3f}'.format(i, avg_loss, train_accuracy, valid_accuracy))

print('\n\nFinished Training\n')
train_logs.printlog()
train_logs.write_csv(os.path.join('experiment/',experiment+'.csv'))
