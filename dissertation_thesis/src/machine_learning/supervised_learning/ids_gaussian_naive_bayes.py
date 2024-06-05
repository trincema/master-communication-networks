import numpy as np 
import matplotlib.pyplot as plt 
import time
# Importing libraries and splitting the dataset 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn.metrics import ConfusionMatrixDisplay

from ids_kdd_cup99_data_preprocessing import KddCup99DataProcessing
from ids import IntrusionDetectionSystem

# Apply various machine learning classification algorithms such as Support Vector Machines,
# Random Forest, Naive Bayes, Decision Tree, Logistic Regression to create different models.
class GaussianNaiveBayesClassifier(IntrusionDetectionSystem):
    def __init__(self) -> None:
        super().__init__()
        print("=== Gaussian Naive Bayes Classifier ===")
        self.classifier = GaussianNB()

    def train(self, data_train, labels_train):
        start_time = time.time()
        self.classifier.fit(data_train, labels_train.values.ravel())
        end_time = time.time()
        print("Training time: ", end_time-start_time)
        print("Train score for NaiveBayes is:", self.classifier.score(data_train, labels_train))

    def test(self, data_test, labels_test):
        start_time = time.time()
        predictions = self.classifier.predict(data_test)
        end_time = time.time()
        print("Testing time: ", end_time-start_time)
        print("Test score for NaiveBayes is:", self.classifier.score(data_test, labels_test))
        return predictions

    def confussionMatrix(self, labels_test, predictions):
        # Get unique classes from labels_test and predictions
        unique_labels = np.unique(labels_test.values.ravel())
        unique_predictions = np.unique(predictions)
        # Combine unique classes from both labels_test and predictions
        all_classes = np.unique(np.concatenate((unique_labels, unique_predictions)))
        # Create a dictionary mapping all possible classes to their indices
        class_mapping = {cls: i for i, cls in enumerate(all_classes)}
        # Creating empty confusion matrix with all possible classes
        conf_matrix = np.zeros((len(all_classes), len(all_classes)))

        # Fill confusion matrix with actual values
        for true_label, pred_label in zip(labels_test.values.ravel(), predictions):
            true_idx = class_mapping[true_label]
            pred_idx = class_mapping[pred_label]
            conf_matrix[true_idx, pred_idx] += 1

        # Displaying the confusion matrix with all unique classes
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=all_classes)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

dataProcessing = KddCup99DataProcessing()
dataProcessing.modelling()
dataProcessing.splitTrainTestData()

classifier = GaussianNaiveBayesClassifier()
classifier.train(dataProcessing.data_train, dataProcessing.labels_train)
predictions = classifier.test(dataProcessing.data_test, dataProcessing.labels_test)
classifier.confussionMatrix(dataProcessing.labels_train, predictions)
classifier.save_trained_classifier(classifier.classifier, 'gaussian_naive_bayes_classifier')