import pandas as pd
import matplotlib.pyplot as plt

# Load the KDD Cup 1999 dataset
# You can download the dataset from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# For this example, let's assume you have a CSV file with the dataset named "kddcup.data_10_percent"
dataset_path = "kddcup.testdata.unlabeled.gz"
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "attack_type"]

# Load the dataset into a pandas DataFrame
df = pd.read_csv(dataset_path, names=column_names)

# Plot the distribution of attack categories
attack_counts = df["attack_type"].value_counts()
attack_counts.plot(kind="bar", figsize=(10, 6), color="skyblue")
plt.title("Distribution of Attack Categories in KDD Cup 1999")
plt.xlabel("Attack Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.show()
