"""
Helper utilitas kompatibilitas lokal untuk praktikum JS14 CNN.
Memastikan dataset CIFAR-10 dan MNIST dapat dimuat secara offline dan bebas error di lingkungan lokal.
"""
import os, sys, pickle
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def patch_datasets():
    try:
        from tensorflow.keras import datasets
        
        # Patch CIFAR-10
        orig_cifar = datasets.cifar10.load_data
        def local_cifar_load():
            candidates = [
                'cifar-10-batches-py',
                'JS14/cifar-10-batches-py',
                '../JS14/cifar-10-batches-py',
                os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')
            ]
            base = None
            for c in candidates:
                if os.path.exists(c) and os.path.exists(os.path.join(c, 'data_batch_1')):
                    base = c
                    break
            if base is None:
                return orig_cifar()
                
            def load_batch(fpath):
                with open(fpath, 'rb') as f:
                    d = pickle.load(f, encoding='bytes')
                    data = d[b'data']
                    labels = d[b'labels']
                data = data.reshape(len(data), 3, 32, 32)
                return data, labels

            x_train = np.empty((50000, 3, 32, 32), dtype='uint8')
            y_train = np.empty((50000,), dtype='uint8')
            for i in range(1, 6):
                fpath = os.path.join(base, f'data_batch_{i}')
                x_train[(i-1)*10000:i*10000], y_train[(i-1)*10000:i*10000] = load_batch(fpath)

            x_test, y_test = load_batch(os.path.join(base, 'test_batch'))
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)
            y_train = np.reshape(y_train, (len(y_train), 1))
            y_test = np.reshape(y_test, (len(y_test), 1))
            return (x_train, y_train), (x_test, y_test)

        datasets.cifar10.load_data = local_cifar_load
        
        # Patch MNIST
        orig_mnist = datasets.mnist.load_data
        def local_mnist_load():
            candidates = [
                'mnist.npz',
                'JS14/mnist.npz',
                'JS13/mnist.npz',
                '../JS14/mnist.npz',
                os.path.join(os.path.dirname(__file__), 'mnist.npz')
            ]
            mnist_path = None
            for c in candidates:
                if os.path.exists(c):
                    mnist_path = c
                    break
            if mnist_path is None:
                return orig_mnist()
            with np.load(mnist_path, allow_pickle=True) as f:
                x_train, y_train = f['x_train'], f['y_train']
                x_test, y_test = f['x_test'], f['y_test']
            return (x_train, y_train), (x_test, y_test)

        datasets.mnist.load_data = local_mnist_load
    except Exception as e:
        pass

patch_datasets()

