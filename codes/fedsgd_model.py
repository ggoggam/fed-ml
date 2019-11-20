import tensorflow as tf

class Model(tf.keras.Model):

    """
    Simple 2-Layer Dense Neural Network for MNIST.
    
    Generalized model for client use.
    This model will be used for control as well: server-side execution

        - Layer 1 : Dense Layer with model.dense_size / ReLU activation
        - Layer 2 : Dense Layer with model.num_classes / Softmax activation

    Inherits from tf.keras.Model

        - Trainable Variables
            - model.dense1
            - model.dense2
        
        - Optimizer
            - model.optimizer (Default : SGD)

        - Constants
            - model.batch_size : Initialization input by user (Default : 64)
            - model.num_classes : Initialization input by user (Default : 10)
    """
    
    def __init__(self, num_features, num_classes):

        super(Model, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Hyperparameters
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
        self.dense_size = 200

        # Layers
        self.W1 = tf.Variable(tf.random.normal([self.num_features, self.dense_size], stddev=0.1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random.normal([self.dense_size], stddev=0.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.normal([self.dense_size, self.num_classes], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.normal([self.num_classes], stddev=0.1, dtype=tf.float32))

    def call(self, inputs):

        out1 = tf.nn.relu(inputs @ self.W1 + self.b1)
        out2 = tf.nn.softmax(out1 @ self.W2 + self.b2)

        return out2

    def loss(self, logits, labels):

        labels_one_hot = tf.one_hot(tf.cast(labels, dtype=tf.uint8), self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits)

        return tf.reduce_mean(loss)