import tensorflow as tf
import numpy as np

class ClientModel(tf.keras.Model):

    """
    Simple 1-Layer Dense Neural Network for MNIST (upto Client Split).
    
    Generalized model for client use.
    
    Inherits from tf.keras.Model

        - Trainable Variables
            - model.dense1
        
        - Optimizer
            - model.optimizer (Default : SGD)

        - Constants
            - model.batch_size : Initialization input by user (Default : 64)
            - model.num_classes : Initialization input by user (Default : 10)
    """
   
    def __init__(self, num_features, num_classes):

        super(ClientModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Hyperparameters
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
        self.dense_size = 200

        # Layers
        self.W = tf.Variable(tf.random.normal([self.num_features, self.dense_size], stddev=0.1, dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal([self.dense_size], stddev=0.1, dtype=tf.float32))

    def call(self, inputs):

        logits = tf.nn.relu(inputs @ self.W + self.b)

        return logits

    def loss(self, logits, labels):

        labels_one_hot = tf.one_hot(tf.cast(labels, dtype=tf.uint8), self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits)

        return tf.reduce_mean(loss)

class ServerModel(tf.keras.Model):

    """
    Simple 1-Layer Dense Neural Network for MNIST (from Client Split)
    
    Generalized model for server use.
    
    Inherits from tf.keras.Model

        - Trainable Variables
            - model.dense1
        
        - Optimizer
            - model.optimizer (Default : SGD)

        - Constants
            - model.batch_size : Initialization input by user (Default : 64)
            - model.num_classes : Initialization input by user (Default : 10)
    """
   
    def __init__(self, num_features, num_classes):

        super(ServerModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Hyperparameters
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
        self.dense_size = 200

        # Layers
        self.W = tf.Variable(tf.random.normal([self.dense_size, self.num_classes], stddev=0.1, dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal([self.num_classes], stddev=0.1, dtype=tf.float32))

    def call(self, inputs):

        logits = tf.nn.softmax(inputs @ self.W + self.b)

        return logits

    def loss(self, logits, labels):

        labels_one_hot = tf.one_hot(tf.cast(labels, dtype=tf.uint8), self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits)

        return tf.reduce_mean(loss)

