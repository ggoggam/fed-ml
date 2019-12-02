import tensorflow as tf
import numpy as np

import vanilla

import fedavg
import splitnn

def main():
    mode = 'FEDAVG'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if mode == 'FEDAVG':    
        # Load Data and Normalize    
        x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
        x_test = x_test.reshape(10000, 784).astype('float32') / 255.0 
        
        # Create Clients
        clients = fedavg.create_clients(x_train, y_train, x_test, y_test, 100)

        # Create Server
        server = fedavg.Server(clients)

        # Update
        server.update()

    if mode == 'SPLITNN':
        # Load Data and Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Create Client
        client = splitnn.Client(x_train[:1000], y_train[:1000])

        # Create Server
        server = splitnn.Server(client)
        
        # Update
        num_rounds = 25
        for i in range(num_rounds):
            g_client = server.train()
            client.update(g_client)
            print("TEST ACCURACY: %.2f %%" % (server.test(tf.expand_dims(x_test[:200], 3), y_test[:200]) * 100))


    
if __name__ == '__main__':
    main()