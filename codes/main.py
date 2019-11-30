import tensorflow as tf
import numpy as np

import vanilla

from fedavg import *

def main():
    # Load Data and Noramalize
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0 
    
    # Create Clients
    clients = create_clients(x_train, y_train, x_test, y_test, 100)

    # Create Server
    server = Server(clients)
    server.update()
    
if __name__ == '__main__':
    main()