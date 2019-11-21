import gzip
import numpy as np
import tensorflow as tf

import dataset
import vanilla

import fedavg

def main():

    train_input_path = '../MNIST_data/train-images-idx3-ubyte.gz'
    train_label_path = '../MNIST_data/train-labels-idx1-ubyte.gz'
    test_input_path = '../MNIST_data/t10k-images-idx3-ubyte.gz'
    test_label_path = '../MNIST_data/t10k-labels-idx1-ubyte.gz'

    tr_i, tr_l = dataset.get_data(train_input_path, train_label_path, 60000)
    tt_i, tt_l = dataset.get_data(test_input_path, test_label_path, 10000)

    ### Shuffle Dataset (IID Assumption)
    train_inputs, train_labels, test_inputs, test_labels = dataset.shuffle_dataset(tr_i, tr_l, tt_i, tt_l)

    # Control for Comparison (IID)
    # control_model_shuffled = vanilla.Model(784, 10)
    # vanilla.train(control_model_shuffled, train_inputs, train_labels, 100, 20)
    # control_shuffled_acc = vanilla.test(control_model_shuffled, test_inputs, test_labels)
    # print("Control Accuracy (IID): %.2f %%" % (control_shuffled_acc * 100))

    ### Sort the Dataset for Creating Biased Dataset (Non-IID Assumption)
    train_inputs, train_labels, test_inputs, test_labels = dataset.sort_dataset(tr_i, tr_l, tt_i, tt_l)

    # Control for Comparison (Non-IID)
    # control_model_sorted = vanilla.Model(784, 10)
    # vanilla.train(control_model_sorted, train_inputs, train_labels, 100, 20)
    # control_sorted_acc = vanilla.test(control_model_sorted, test_inputs, test_labels)
    # print("Control Accuracy (Non-IID): %.2f %%" % (control_sorted_acc * 100))

    # FedSGD
    fed_server = fedavg.server(train_inputs, train_labels, 100)
    fed_acc = fed_server.update(test_inputs, test_labels)
    print("FedSGD Accuracy : %.2f %%" % (fed_acc * 100))
    
if __name__ == '__main__':
    main()