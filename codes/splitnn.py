import tensorflow as tf
import numpy as np

# Control Model (Single Entity)
def control(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), (1,1), padding='valid', activation='relu'),
                                tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='valid', activation='relu'),
                                tf.keras.layers.MaxPool2D((2,2)),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(10, activation='softmax')])
    model.build((None, 28, 28, 1))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'], optimizer=tf.keras.optimizers.SGD(1e-2))
    model.fit(np.expand_dims(x_train[:1000], 3), y_train[:1000], batch_size=50, epochs=25, verbose=0)
    model.evaluate(np.expand_dims(x_test[:200], 3), y_test[:200])

class Client(tf.keras.Model):

    def __init__(self, x_train, y_train):
        super(Client, self).__init__()

        # Inputs
        self.num_samples = x_train.shape[0]
        self.train_inputs = tf.expand_dims(x_train, 3)
        self.train_labels = y_train

        # Hyperparameter
        self.batch_size = 50

        # Client Model (Split Layer)
        self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), (1,1), padding='valid', activation='relu')])

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)    

    def call(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def send(self, i):
        start, end = i * self.batch_size, (i+1) * self.batch_size

        logits = self.call(self.train_inputs[start:end])
        labels = self.train_labels[start:end]
        return logits, labels
    
    def update(self, g_client):

        self.optimizer.apply_gradients(zip(g_client, self.trainable_variables))


class Server(tf.keras.Model):

    def __init__(self, client):
        super(Server, self).__init__()

        self.num_samples = client.num_samples
        self.client = client
        self.batch_size = client.batch_size

        # Model
        self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3,3), (1,1), padding='valid', activation='relu'),
                                          tf.keras.layers.MaxPool2D((2,2)),
                                          tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(10, activation='softmax')])
        # Loss
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy() 

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(1e-2)

    def call(self, split_logits):
        return self.model(split_logits)

    def loss(self, logits, labels):
        return self.loss_function(labels, logits)

    def train(self):

        for i in range(self.num_samples // self.batch_size):

            with tf.GradientTape(persistent=True) as tape:
                logits, labels = self.client.send(i)
                final_logits = self.call(logits)
                l = self.loss(final_logits, labels)

            g_server = tape.gradient(l, self.trainable_variables)
            g_client = tape.gradient(l, self.client.trainable_variables)

            self.optimizer.apply_gradients(zip(g_server, self.trainable_variables))

        return g_client    


    def test(self, inputs, labels):
        split_logits = self.client.call(inputs)
        logits = self.call(split_logits)

        predictions = np.argmax(logits, axis=1)
        acc = np.mean(predictions == labels)

        return acc