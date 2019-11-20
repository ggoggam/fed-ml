import numpy as np
import tensorflow as tf

import dataset
import fedsgd_model

"""
Federated SGD Algorithm Demo

    - Reference Paper
        "Communication-Efficient Learning of Deep Networks from Decentralized Data"
        https://arxiv.org/pdf/1602.05629.pdf
"""

class client:
    
    """
    class client
    
    Individual client has biased dataset of possibly different sizes. 
    """

    def __init__(self, inputs, labels, num_epochs=20, num_classes=10):
        
        # Client Data
        self.inputs = inputs
        self.labels = labels

        self.num_samples = self.inputs.shape[0]
        self.num_features = self.inputs.shape[1]
        self.num_classes = num_classes

        # Hyperparameters
        self.num_epochs = num_epochs
        self.batch_size =  self.num_samples // 12
        if self.batch_size == 0 : self.batch_size = 1

        # Individual Client Model
        self.model = fedsgd_model.Model(self.num_features, self.num_classes)

    def client_update(self):

        n = self.batch_size
        num_batches = self.num_samples // n

        for epoch in range(self.num_epochs):
        
            for i in range(num_batches):
                start, end = i*n, (i+1)*n
                
                with tf.GradientTape() as tape:
                    logits = self.model.call(self.inputs[start:end])
                    loss = self.model.loss(logits, self.labels[start:end])
        
                grad = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class server:
    
    """
    class server

    A server keeps a list of clients. 
    """

    def __init__(self, clients, num_rounds=100, num_features=784, num_classes=10, dense_size=200, C_fixed=True):

        ### Client List
        self.clients = clients
        self.num_clients = len(clients)
        self.total_samples = sum(c.num_samples for c in self.clients)
        self.C_fixed = C_fixed

        ### Hyperparameters
        self.num_rounds = num_rounds
        self.num_features = num_features
        self.num_classes = num_classes

        self.dense_size = dense_size

        ### Server Weight Initialization
        self.W1 = tf.zeros([self.num_features,self.dense_size])
        self.b1 = tf.zeros([self.dense_size])
        self.W2 = tf.zeros([self.dense_size,self.num_classes])
        self.b2 = tf.zeros([self.num_classes])

    def server_update(self, inputs, labels, print_progress=False):
        
        for t in range(self.num_rounds):
            
            # Create Sample of Participating Clients
            if self.C_fixed:
                p = 1
            else:
                p = 1.2 * np.random.rand()
                if p > 1: p = 1

            m = int(self.num_clients * p)
            sample = np.random.choice(self.clients, m)

            # Client Update on Sample
            # Should be done in parallel, but we skip this parallel implementation
            cnt = 1
            for client in sample:

                # Client Update
                if print_progress: 
                    print('ROUNDS %d / %d...   CLIENT UPDATE %d / %d' % (t+1, self.num_rounds, cnt, m))
                client.client_update()

                cnt += 1
            
            self.W1 = tf.zeros([self.num_features,self.dense_size])
            self.b1 = tf.zeros([self.dense_size])
            self.W2 = tf.zeros([self.dense_size,self.num_classes])
            self.b2 = tf.zeros([self.num_classes])

            cnt = 1
            for client in self.clients:

                # Server Update
                weights = client.model.get_weights()
                client_weight = client.num_samples / self.total_samples

                self.W1 += weights[0] * client_weight
                self.b1 += weights[1] * client_weight
                self.W2 += weights[2] * client_weight
                self.b2 += weights[3] * client_weight

                cnt += 1

            acc = self.test(inputs, labels)
            print('======== ROUND %d =========' % (t))
            print('SERVER UPDATE Accuracy : %.2f %%' % (acc * 100))
        
        return acc
            
    def server_call(self, inputs):

        inputs = tf.cast(inputs, dtype=tf.float32)
        dense = inputs @ self.W1 + self.b1
        logit = dense @ self.W2 + self.b2

        return logit

    def test(self, inputs, labels):
        
        logits = self.server_call(inputs)
        predictions = np.argmax(logits, axis=1)
        acc = np.mean(labels == predictions)

        return acc  

def create_clients(inputs, labels, num_clients):

    n = inputs.shape[0] // num_clients
    shard_ind = tf.random.shuffle(range(num_clients))

    clients = []
    for i in range(num_clients):
        start, end = i*n, (i+1)*n
        
        clients.append(client(inputs=tf.cast(inputs[start:end], dtype=tf.float32), labels=tf.cast(labels[start:end], dtype=tf.float32)))
    
    return clients