from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

# Define the model
model = Sequential()
model.add(Input(shape=(41,)))  # KDD Cup 99 dataset has 41 features

# First hidden layer
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

# Second hidden layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))  # Assuming binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
