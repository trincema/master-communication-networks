from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

# Example input shape: (sequence_length, features)
sequence_length = 100  # Example sequence length
num_features = 41      # Number of features in KDD Cup 99 dataset

# Define the model
model = Sequential()
model.add(Input(shape=(sequence_length, num_features)))

# First Conv1D layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Second Conv1D layer
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten and fully connected layer
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))  # Assuming binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
