import tensorflow as tf
import numpy as np

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
        self.dense1 = tf.keras.layers.Dense(self.dense_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):

        out = self.dense1(inputs)
        logits = self.dense2(out)

        return logits

    def loss(self, logits, labels):

        labels_one_hot = tf.one_hot(tf.cast(labels, dtype=tf.uint8), self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits)

        return tf.reduce_mean(loss)

def train(model, inputs, labels, batch_size, epochs):
    
    for t in range(epochs):

        for i in range(inputs.shape[0]//batch_size):

            start, end = i*batch_size, (i+1)*batch_size

            with tf.GradientTape() as tape:

                logits = model.call(inputs[start:end])
                loss = model.loss(logits, labels[start:end])
            
            grad = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        
def test(model, inputs, labels):

    logits = model.call(inputs)
    predictions = np.argmax(logits, axis=1)
    acc = np.mean(labels == predictions)

    return acc