import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load a sample dataset (replace with your own dataset)
# The dataset should include features related to network traffic.
# For a real-world scenario, consider using datasets like NSL-KDD.
# https://www.unb.ca/cic/datasets/nsl.html
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00368/Friday-WorkingHours-Morning.pcap_ISCX.csv"
df = pd.read_csv(data_url, skipinitialspace=True)

# Data preprocessing
# In a real-world scenario, you may need to clean and preprocess your data more thoroughly.
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop(["Timestamp", "Flow ID", "Source IP", "Destination IP", "Label"], axis=1, inplace=True)
X = df.drop("Label", axis=1)
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the k-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))
