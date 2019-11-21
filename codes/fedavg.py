import tensorflow as tf
import numpy as np

import dataset

def create_clients(inputs, labels, num_clients):

    # Client Data Size
    n = inputs.shape[0] // num_clients
    num_shards = 2
    
    # Client Shard Size
    input_shards, label_shards = dataset.generate_shards(inputs, labels, n//num_shards)
    
    tf.random.shuffle(input_shards)
    tf.random.shuffle(label_shards)

    clients = []
    for i in range(num_clients):
        j = i*num_shards
        input_data = tf.concat([input_shards[j], input_shards[j+1]], axis=0)
        label_data = tf.concat([label_shards[j], label_shards[j+1]], axis=0)
        clients.append(client(inputs=tf.cast(input_data, dtype=tf.float32), labels=tf.cast(label_data, dtype=tf.float32)))
    
    return clients

class client(tf.keras.Model):

    def __init__(self, inputs, labels):
        
        super(client, self).__init__()

        self.inputs = tf.random.shuffle(inputs)
        self.labels = tf.random.shuffle(labels)

        self.num_samples = self.inputs.shape[0]
        self.num_features = self.inputs.shape[1]

        self.num_classes = 10
        self.num_epochs = 20
        self.batch_size = self.num_samples // 20
        self.dense_size = 200

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)

        self.W1 = tf.Variable(tf.zeros([self.num_features, self.dense_size]))
        self.b1 = tf.Variable(tf.zeros([self.dense_size]))
        self.W2 = tf.Variable(tf.zeros([self.dense_size, self.num_classes]))
        self.b2 = tf.Variable(tf.zeros([self.num_classes]))

    def call(self, inputs):

        out = tf.nn.relu(inputs @ self.W1 + self.b1)
        logits = tf.nn.softmax(out @ self.W2 + self.b2)

        return logits

    def loss_function(self, inputs, labels):

        logits = self.call(inputs)
        labels_one_hot = tf.one_hot(tf.cast(labels, dtype=tf.uint8), self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels_one_hot, logits)

        return tf.reduce_mean(loss)

    def train(self):

        for i in range(self.num_samples//self.batch_size):
            start, end = i*self.batch_size, (i+1)*self.batch_size

            with tf.GradientTape() as tape:
                loss = self.loss_function(self.inputs[start:end], self.labels[start:end])
            
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def assign_weights(self, W1, b1, W2, b2):

        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

class server:

    def __init__(self, inputs, labels, num_clients):

        self.num_clients = num_clients
        self.clients = create_clients(inputs, labels, num_clients)

        self.total_samples = sum(c.num_samples for c in self.clients)

        self.num_rounds = 150
        self.num_features = self.clients[0].num_features
        self.num_classes = self.clients[0].num_classes
        self.dense_size = self.clients[0].dense_size

        self.W1 = tf.random.normal([self.num_features,self.dense_size], stddev=0.1, dtype=tf.float32)
        self.b1 = tf.random.normal([self.dense_size], stddev=0.1, dtype=tf.float32)
        self.W2 = tf.random.normal([self.dense_size,self.num_classes], stddev=0.1, dtype=tf.float32)
        self.b2 = tf.random.normal([self.num_classes], stddev=0.1, dtype=tf.float32)
    
    def update(self, inputs, labels):

        for t in range(self.num_rounds):

            C_fix = True

            if C_fix: 
                p = 1
            else:
                p = min(1, 1.2 * np.random.rand())

            m = max(1, int(self.num_clients * p))
            sample = np.random.choice(self.clients, m)
            print(sample)

            for client in sample:
                client.assign_weights(self.W1, self.b1, self.W2, self.b2)
                client.train()

            self.W1 = tf.zeros([self.num_features, self.dense_size])
            self.b1 = tf.zeros([self.dense_size])
            self.W2 = tf.zeros([self.dense_size, self.num_classes])
            self.b2 = tf.zeros([self.num_classes])
            for client in self.clients:
                client_weight = client.num_samples/self.total_samples

                self.W1 += client.W1 * client_weight
                self.b1 += client.b1 * client_weight
                self.W2 += client.W2 * client_weight
                self.b2 += client.b2 * client_weight

            acc = self.test(inputs, labels)
            print('ROUND %d / %d UPDATE ACCURACY : %.2f %%' % (t+1, self.num_rounds, acc * 100))
            
        return acc

    def call(self, inputs):

        inputs = tf.cast(inputs, dtype=tf.float32)
        dense = inputs @ self.W1 + self.b1
        logit = dense @ self.W2 + self.b2

        return logit   
    
    def test(self, inputs, labels):
        
        logits = self.call(inputs)
        predictions = np.argmax(logits, axis=1)
        acc = np.mean(labels == predictions)

        return acc  