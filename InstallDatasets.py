import numpy as np
import torchvision.datasets as datasets

mnist = datasets.MNIST(root='./data', train=False, download=True)
images = mnist.data.numpy()
labels = mnist.targets.numpy()

flat_images = images.reshape(-1, 784).astype(np,int8)
labels = labels.astype(np.uint8)

flat_images.tofile("mnist_images_10k.bin")
labels.tofile("mnist_labels_10k.bin")
print("Export complete: 10,000 images written (7,840,000 bytes).")
