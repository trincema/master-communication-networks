# Author: Emanuel TRINC
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
print(digits.data)
print("images: " + str(len(digits.images)) +
    " of size (" + str(len(digits.images[0])) +
    ", " + str(len(digits.images[0][0])) + ")")
print("data set size: rows = " + str(len(digits.data)) + " columns = " + str(len(digits.data[0])))
print("target: " + str(digits.target) + " size: " + str(len(digits.target)))

# PERFORM TRAINING
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

print(digits.images)
print("n_samples = " + str(len(digits.images)))