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
        self.dense_size = 200
        
        model = 'DENSE'

        # Layers
        if model == 'DENSE':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(self.dense_size, activation='relu'),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
        if model == 'CNN':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), (1,1), padding='valid', activation='relu'),
                tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='valid', activation='relu'),
                tf.keras.layers.MaxPool2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)

        # Loss
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        one_hot = tf.one_hot(labels, self.num_classes)
        l = self.loss_function(one_hot, logits)
        return tf.reduce_mean(l)

def train(model, inputs, labels, batch_size, epochs):
    
    for _ in range(epochs):

        for i in range(inputs.shape[0]//batch_size):
            start, end = i*batch_size, (i+1)*batch_size

            with tf.GradientTape() as tape:
                l = model.loss(model.call(inputs[start:end]), labels[start:end])
            
            g = tape.gradient(l, model.trainable_variables)
            model.optimizer.apply_gradients(zip(g, model.trainable_variables))
        
def test(model, inputs, labels):

    logits = model.call(inputs)
    predictions = np.argmax(logits, axis=1)
    acc = np.mean(labels == predictions)

    return acc
