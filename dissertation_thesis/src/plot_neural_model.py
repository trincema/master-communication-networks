import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('out/shallow_model.h5')

# Choose a sample input from your dataset
sample_input = X_test[0]  # Replace with your own sample input

# Create a function to extract activations from intermediate layers
activation_model = tf.keras.models.Model(inputs=model.input, 
                                          outputs=[layer.output for layer in model.layers])

# Get activations for the sample input
activations = activation_model.predict(sample_input.reshape(1, -1))

# Visualize activations for each layer
for layer_activation, layer_name in zip(activations, model.layers):
    plt.figure()
    plt.matshow(layer_activation[0, :, :])  # Assuming 2D activations
    plt.title(layer_name)
    plt.show()
