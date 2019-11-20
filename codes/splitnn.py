import numpy as np
import tensorflow as tf

import splitnn_model

class client:

    def __init__(self, inputs, labels, num_epochs=20, num_classes=10):

        # Client Data
        self.inputs = inputs
        self.labels = labels

        self.num_samples = self.inputs.shape[0]
        self.num_features = self.inputs.shape[1]
        self.num_classes = num_classes

        # Hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = self.num_samples // 12
        if self.batch_size == 0: self.batch_size = 1

        # Individual Client Model
        self.model = splitnn_model.ClientModel(self.num_classes)

    def client_call(self):

        logits = self.model.call(self.inputs)

        return logits
    
    def client_backprop(self):

        for i in range(self.num_samples//self.batch_size):

            start, end = i*self.batch_size, (i+1)*self.batch_size
        
            with tf.GradientTape() as tape:
                logits = self.model.call(self.inputs[start:end])
                loss =self.model.loss(logits, self.labels[start:end])

            grad = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients()

        