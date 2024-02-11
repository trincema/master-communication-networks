import joblib
# Neural Network imports
import tensorflow as tf
from keras.models import load_model

class IntrusionDetectionSystem:
    def save_trained_classifier(self, classifier, file_name):
        # Save the trained model to a file
        joblib.dump(classifier, 'out/{0}.joblib'.format(file_name))
    
    def load_trained_classifier(self, classifier, file_name):
        # Load the saved model from the file
        classifier = joblib.load('{0}.joblib'.format(file_name))

    def save_trained_model(self, model, file_name):
        """
        Save the trained model to a file to be able to load it back again without the need to retrain.
        """
        # Save the trained model to a file (Python-compatible)
        model.save("out/{0}.h5".format(file_name))

        # Convert the model to TensorFlow Lite format (Android)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.shallow_model)
        tflite_model = converter.convert()
        # Save the TensorFlow Lite model to a file
        with open('out/{0}.tflite'.format(file_name), 'wb') as f:
            f.write(tflite_model)
    
    def load_trained_model(self, model, file_name):
        """
        """
        model = load_model("{0}.h5".format(file_name))