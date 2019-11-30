import tensorflow as tf
import numpy as np
import random

def build_dict(y_train, y_test):

    train_dict, test_dict = {}, {}
    for label in range(10):
      train_dict[label] = np.argwhere(label==y_train).reshape(-1)
      test_dict[label] = np.argwhere(label==y_test).reshape(-1)

    return train_dict, test_dict

def create_clients(x_train, y_train, x_test, y_test, num_clients):

    """
      Builds a list of Non-IID clients for the demo
        - Generates dictionaries of train and test dataset indices according to labels
        - Based on dictionaries, pick two random labels with replacement
        - Pick 100 random samples from the picked labels for train and test set
    """
    # Client Data Size
    client_size = x_train.shape[0] // num_clients
    shard_size = client_size // 2

    # Build Dictionaries
    train_dict, test_dict = build_dict(y_train, y_test)

    clients = []
    for i in range(num_clients):
        if (i % 10 == 0): print("Generating %d-th Clients ..." % (i+10))
        labels = np.random.choice(10, 2)

        train_ind = [np.random.choice(train_dict[l], shard_size) for l in labels]
        train_ind = [i for l in train_ind for i in l]
        train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
        train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)

        test_ind = [np.random.choice(test_dict[l], shard_size) for l in labels]
        test_ind = [i for l in test_ind for i in l]
        test_inputs = tf.cast(tf.gather(x_test, test_ind), tf.float32)
        test_labels = tf.cast(tf.gather(y_test, test_ind), tf.uint8)

        clients.append(Client(train_inputs, train_labels, test_inputs, test_labels))
    
    return clients

class Client(tf.keras.Model):

    def __init__(self, x_train, y_train, x_test, y_test):
        super(Client, self).__init__()
        
        # Inputs and Labels
        self.num_samples = x_train.shape[0]

        self.train_inputs, self.train_labels = x_train, y_train
        self.test_inputs, self.test_labels = x_test, y_test

        # Hyperparameters
        self.hidden_size = 200
        self.num_classes = 10
        self.num_epochs = 10
        self.batch_size = self.num_samples // 30

        # Initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.1)

        # Layers (Need to be built prior to call)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu',
                                  kernel_initializer=init, bias_initializer=init),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer=init, bias_initializer=init)
        ])
        self.model.build((None, 784))

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

        # Loss
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        l = self.loss_function(labels, logits)
        return tf.reduce_mean(l)

    def train(self, weights):
        self.model.set_weights(weights)
        for _ in range(self.num_epochs):
            shuffled = tf.random.shuffle(range(self.num_samples))
            inputs = tf.gather(self.train_inputs, shuffled)
            labels = tf.gather(self.train_labels, shuffled)
            for i in range(self.num_samples//self.batch_size):
                start, end = i*self.batch_size, (i+1)*self.batch_size

                with tf.GradientTape() as tape:
                    l = self.loss(self.call(inputs[start:end]), labels[start:end])
                
                g = tape.gradient(l, self.trainable_variables)
                self.optimizer.apply_gradients(zip(g, self.trainable_variables))

        return self.model.get_weights()

    def test(self):
        logits = self.call(self.test_inputs)
        preds = np.argmax(logits, axis=1)
        num_correct = np.sum(self.test_labels == preds)
        return num_correct



class Server:

    def __init__(self, clients):     
        # Clients
        self.clients = clients

        self.num_clients = len(clients)
        self.total_samples = sum(c.num_samples for c in self.clients)

        # Hyperparameters
        self.num_rounds = 50
        self.num_classes = self.clients[0].num_classes
        self.hidden_size = self.clients[0].hidden_size

        self.C_fixed = False

        # Initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.1)

        # Layers (Needs to be built before call)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu',
                                  kernel_initializer=init, bias_initializer=init),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer=init, bias_initializer=init)
        ])
        self.model.build((None, 784))
    
    def update(self):
        for t in range(self.num_rounds):
            # Create Random Sample of Clients
            if self.C_fixed: 
                p = 1
            else:
                p = min(1, 1.1 * random.random())

            m = max(1, int(self.num_clients * p))
            sample = np.random.choice(self.clients, m)
            print('ROUND %d / %d NUM SAMPLES : %d' % (t+1, self.num_rounds, m))
            
            # Update Client Weights (Train) in Sample
            for client in sample:
                weights = client.train(self.model.get_weights())
                client.model.set_weights(weights)

            # Update Server Weights
            server_weights = self.model.get_weights()
            for client in self.clients:
                pk = client.num_samples / self.total_samples

                for sw, w in zip(server_weights, client.model.get_weights()):
                    sw = sw + w * pk

            self.model.set_weights(server_weights)

            # Testing
            acc = self.test()
            print('ROUND %d / %d UPDATE ACCURACY : %.2f %%' % (t+1, self.num_rounds, acc * 100))
            
        return acc

    def call(self, inputs):
        return self.model(inputs)
    
    def test(self):
        correct, samples = 0, 0
        for c in self.clients:
            samples += c.test_inputs.shape[0]
            correct += c.test()
        return correct / samples