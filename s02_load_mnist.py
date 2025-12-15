from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from s01_config import save_fig



# print(mnist.DESCR)
# print(mnist.keys())


## Extract test and training sets from mnist fetch
def create_mnist_test_train():
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test
    
## Short data exploration (X, y, and shape)
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

X_train, X_test, y_train, y_test = create_mnist_test_train()

## Exploration of input features (how they're presented)
# import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    
# some_digit = X_train[0]
# plot_digit(some_digit)
# save_fig("some_digit_plot")
# plt.show()

# print(y_train[0])

## Create a 10x10 image containing the first 100 digits in the training set
# plt.figure(figsize=(9,9))
# for idx, image_data in enumerate(X_train[:100]):
#     plt.subplot(10, 10, idx + 1)
#     plot_digit(image_data)
# plt.subplots_adjust(wspace=0, hspace=0)
# save_fig("100_digits_visualised", tight_layout=False)
# plt.show()