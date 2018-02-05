import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Reduced size
k = 128

# Training data
X = mnist.train.images

# Mean X
mean_X = np.mean(X, axis = 0)

# Mean normalization
X_centered = X - mean_X

# Covariance matrix
sigma = np.dot(X_centered.T, X_centered)/X_centered.shape[0]

# Perform SVD
U, s, V = np.linalg.svd(sigma)

# Select first k eigenvectors of U
U_k = U[:,0:k]

# Compress data
Z = np.dot(X_centered, U_k)

# Reconstruct centered data
X_centered_hat = np.dot(Z, U_k.T)

# Reconstruct raw data
X_hat = X_centered + mean_X

# Calculate loss
loss = np.mean((X - X_hat) ** 2)
print 'Loss (training data) is', loss

# Visualize original and reconstructed images using training images
n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))

for i in range(n):
    origin_images = X[i:i+n]
    reconstructed_images = X_hat[i:i+n]
    # Display original images
    for j in range(n):
        # Draw original digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            origin_images[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw reconstructed digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            reconstructed_images[j].reshape([28, 28])
print 'Original Images'
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print 'Reconstructed Images'
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

# TESTING
X_test = mnist.test.images
X_test_centered = X_test - mean_X
Z_test = np.dot(X_test_centered, U_k)
X_test_centered_hat = np.dot(Z_test, U_k.T)
X_test_hat = X_test_centered_hat + mean_X

# Calculate loss of test data
loss_test = np.mean((X_test - X_test_hat) ** 2)
print 'Loss (test data) is', loss_test

# Visualize original and reconstructed images using test images
n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))

for i in range(n):
    origin_test_images = X_test[i:i+n]
    reconstructed_test_images = X_test_hat[i:i+n]
    # Display original images
    for j in range(n):
        # Draw original digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            origin_test_images[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw reconstructed digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            reconstructed_test_images[j].reshape([28, 28])

print 'Original Test Images'
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print 'Reconstructed Test Images'
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

