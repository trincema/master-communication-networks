import numpy as np 
import matplotlib.pyplot as plt 
import time
# Importing libraries and splitting the dataset 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Neural Network imports
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.models import load_model
# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn.metrics import ConfusionMatrixDisplay

from ids_kdd_cup99_data_preprocessing import KddCup99DataProcessing
from ids import IntrusionDetectionSystem

class DeepNeuralNetworksModel(IntrusionDetectionSystem):
    def __init__(self, data_train) -> None:
        super().__init__()
        print("=== Deep Neural Network Model ===")
        # Create a neural network model
        self.deep_model = Sequential([
            Dense(1024, input_dim=32, activation='relu'),
            Dropout(0.01),
            Dense(768, activation='relu'),
            Dropout(0.01),
            Dense(512, activation='relu'),
            Dropout(0.01),
            Dense(256, activation='relu'),
            Dropout(0.01),
            Dense(128, activation='relu'),
            Dropout(0.01),
            Dense(5, activation='softmax')
        ])
        self.deep_model.compile(loss ='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        tf.keras.utils.plot_model(self.deep_model, to_file="nn_deep_model.png", show_shapes=True)

    def train(self, data_train, labels_train):
        """
        Train the Deep Neural Network model.
        """
        start_time = time.time()
        self.history = self.deep_model.fit(data_train, labels_train.values.ravel(), epochs=10, batch_size=32)
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        # Evaluate the model on the training data
        train_loss, train_accuracy = self.deep_model.evaluate(data_train, labels_train)
        print("Train loss for Neural Network Deep is:", train_loss)
        print("Train accuracy for Neural Network Deep is:", train_accuracy)

    def test(self, data_test, labels_test):
        """
        Evaluate the model on the test set.
        """
        start_time = time.time()
        test_loss, test_accuracy = self.deep_model.evaluate(data_test, labels_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        print(f'Test loss: {test_loss}')
        print(f'Test accuracy: {test_accuracy}')
        return test_loss

    def confussion_matrix(self, labels_test, predictions):
        pass
    
    def load_trained_model(self, file_name):
        """
        """
        # Load the saved model back
        self.deep_model = load_model("{0}.h5".format(file_name))

dataProcessing = KddCup99DataProcessing()
dataProcessing.modelling()
dataProcessing.splitTrainTestData()

model = DeepNeuralNetworksModel(dataProcessing.data_train)
model.train(dataProcessing.data_train, dataProcessing.labels_train)
predictions = model.test(dataProcessing.data_test, dataProcessing.labels_test)
model.save_trained_model(model.deep_model, 'deep_model')