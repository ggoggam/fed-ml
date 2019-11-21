import tensorflow as tf
import numpy as np
import gzip

def get_data(inputs_path, labels_path, num_examples):

    with gzip.open(inputs_path, 'rb') as f1, gzip.open(labels_path, 'rb') as f2:
        X = np.frombuffer(f1.read(16 + num_examples * 784), dtype=np.uint8, offset=16).reshape(num_examples, 784) / 255.0
        Y = np.frombuffer(f2.read(8 + num_examples), dtype=np.uint8, offset=8)

    return X, Y

def shuffle_dataset(tr_i, tr_l, tt_i, tt_l):

    train_shuffled_arg = tf.random.shuffle(range(tr_i.shape[0]))

    train_inputs, train_labels = tf.gather(tr_i, train_shuffled_arg), tf.gather(tr_l, train_shuffled_arg)
    test_inputs, test_labels = tt_i, tt_l

    train_inputs, train_labels = tf.cast(train_inputs, dtype=tf.float32), tf.cast(train_labels, dtype=tf.uint8)
    test_inputs, test_labels = tf.cast(test_inputs, dtype=tf.float32), tf.cast(test_labels, dtype=tf.uint8)

    return train_inputs, train_labels, test_inputs, test_labels

def sort_dataset(tr_i, tr_l, tt_i, tt_l):

    train_sort_arg, test_sort_arg = np.argsort(tr_l), np.argsort(tt_l)

    train_inputs, train_labels = tf.gather(tr_i, train_sort_arg), tf.gather(tr_l, train_sort_arg)
    test_inputs, test_labels = tf.gather(tt_i, test_sort_arg), tf.gather(tt_l, test_sort_arg)
    
    train_inputs, train_labels = tf.cast(train_inputs, dtype=tf.float32), tf.cast(train_labels, dtype=tf.uint8)
    test_inputs, test_labels = tf.cast(test_inputs, dtype=tf.float32), tf.cast(test_labels, dtype=tf.uint8)

    return train_inputs, train_labels, test_inputs, test_labels

def generate_shards(tr_i, tr_l, shard_size):

    input_shards, label_shards = [], []

    for i in range(tr_i.shape[0]//shard_size):
        start, end = i*shard_size, (i+1)*shard_size

        input_shards.append(tr_i[start:end])
        label_shards.append(tr_l[start:end])

    return input_shards, label_shards
