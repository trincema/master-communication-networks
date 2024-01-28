import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time
# Importing libraries and splitting the dataset 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm

# Step 1 – Data Preprocessing:

# Reading features list from ‘kddcup.names’ file. 
with open("data_sets/kddcup.names", 'r') as f:
	print(f.read())

# Appending columns to the dataset and adding a new column name ‘target’ to the dataset.
cols ="""duration, 
	protocol_type, 
	service, 
	flag, 
	src_bytes, 
	dst_bytes, 
	land, 
	wrong_fragment, 
	urgent, 
	hot, 
	num_failed_logins, 
	logged_in, 
	num_compromised, 
	root_shell, 
	su_attempted, 
	num_root, 
	num_file_creations, 
	num_shells, 
	num_access_files, 
	num_outbound_cmds, 
	is_host_login, 
	is_guest_login, 
	count, 
	srv_count, 
	serror_rate, 
	srv_serror_rate, 
	rerror_rate, 
	srv_rerror_rate, 
	same_srv_rate, 
	diff_srv_rate, 
	srv_diff_host_rate, 
	dst_host_count, 
	dst_host_srv_count, 
	dst_host_same_srv_rate, 
	dst_host_diff_srv_rate, 
	dst_host_same_src_port_rate, 
	dst_host_srv_diff_host_rate, 
	dst_host_serror_rate, 
	dst_host_srv_serror_rate, 
	dst_host_rerror_rate, 
	dst_host_srv_rerror_rate"""

columns = []
for c in cols.split(', '):
	if(c.strip()):
		columns.append(c.strip())
columns.append('target')
print(len(columns))

# Reading the ‘training_attack_types’ file.
with open("data_sets/training_attack_types", 'r') as f:
	print(f.read())

# Creating a dictionary of attack_types
attacks_types = { 
	'normal': 'normal',
	'back': 'dos',
	'buffer_overflow': 'u2r',
	'ftp_write': 'r2l',
	'guess_passwd': 'r2l',
	'imap': 'r2l',
	'ipsweep': 'probe',
	'land': 'dos',
	'loadmodule': 'u2r',
	'multihop': 'r2l',
	'neptune': 'dos',
	'nmap': 'probe',
	'perl': 'u2r',
	'phf': 'r2l',
	'pod': 'dos',
	'portsweep': 'probe',
	'rootkit': 'u2r',
	'satan': 'probe',
	'smurf': 'dos',
	'spy': 'r2l',
	'teardrop': 'dos',
	'warezclient': 'r2l',
	'warezmaster': 'r2l',
} 

# Reading the dataset(‘kddcup.data_10_percent.gz’) and adding Attack Type feature in the training
# dataset where attack type feature has 5 distinct values i.e. dos, normal, probe, r2l, u2r.
path = "data_sets/kddcup.data_10_percent.gz"
df = pd.read_csv(path, names = columns)
# Adding Attack Type column 
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
df.head()

# Shape of dataframe and getting data type of each feature 
df.shape

# Finding missing values of all features.
df.isnull().sum()
print(df.columns)

# Finding categorical features 
num_cols = df._get_numeric_data().columns
cate_cols = list(set(df.columns)-set(num_cols))
cate_cols.remove('target')
cate_cols.remove('Attack Type')
print(cate_cols)

# Data Correlation – Find the highly correlated variables using heatmap and ignore them for analysis.
df = df.dropna(axis=1) # drop columns with NaN
# keep columns where there are more than 1 unique values
#df = df.loc[:, df.nunique() > 1]
# Keep only numeric columns
numeric_df = df.select_dtypes(include='number')
# Calculate correlation matrix
corr = numeric_df.corr()
# Plot heatmap
plt.figure(figsize =(15, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.show()


# This variable is highly correlated with num_compromised and should be ignored for analysis. 
#(Correlation = 0.9938277978738366) 
df.drop('num_root', axis = 1, inplace = True) 

# This variable is highly correlated with serror_rate and should be ignored for analysis. 
#(Correlation = 0.9983615072725952) 
df.drop('srv_serror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9947309539817937) 
df.drop('srv_rerror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with srv_serror_rate and should be ignored for analysis. 
#(Correlation = 0.9993041091850098) 
df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9869947924956001) 
df.drop('dst_host_serror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9821663427308375) 
df.drop('dst_host_rerror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9851995540751249) 
df.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True) 

# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9865705438845669) 
df.drop('dst_host_same_srv_rate', axis = 1, inplace = True) 

# Feature Mapping – Apply feature mapping on features such as : ‘protocol_type’ & ‘flag’. 
# protocol_type feature mapping 
pmap = {'icmp':0, 'tcp':1, 'udp':2} 
df['protocol_type'] = df['protocol_type'].map(pmap)  
# flag feature mapping 
fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
df['flag'] = df['flag'].map(fmap) 

# Remove irrelevant features such as ‘service’ before modelling 
df.drop('service', axis = 1, inplace = True) 

# Step 2 – Modelling

# Splitting the dataset 
df = df.drop(['target', ], axis = 1) 
print(df.shape) 

# Target variable and train set 
y = df[['Attack Type']] 
X = df.drop(['Attack Type', ], axis = 1) 

sc = MinMaxScaler() 
X = sc.fit_transform(X) 

# Split test and train data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print(X_train.shape, X_test.shape) 
print(y_train.shape, y_test.shape) 

# Apply various machine learning classification algorithms such as Support Vector Machines,
# Random Forest, Naive Bayes, Decision Tree, Logistic Regression to create different models.

# Python implementation of Gaussian Naive Bayes Classifier
# Training
clfg = GaussianNB() 
start_time = time.time() 
clfg.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 

# Testing 
start_time = time.time()
y_test_pred = clfg.predict(X_train)
end_time = time.time()
print("Testing time: ", end_time-start_time)

print("Train score for NaiveBayes is:", clfg.score(X_train, y_train)) 
print("Test score for NaiveBayes is:", clfg.score(X_test, y_test)) 

# Display Confussion Matrix
# Create confusion matrix
conf_matrix = confusion_matrix(y_train, y_test_pred)
# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Python Implementation of Decision Tree Classifier
clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4) 
start_time = time.time() 
clfd.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 

start_time = time.time() 
y_test_pred = clfd.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 

print("Train score for DecisionTree is:", clfd.score(X_train, y_train)) 
print("Test score for DecisionTree is:", clfd.score(X_test, y_test)) 

# Python code implementation of Random Forest

from sklearn.ensemble import RandomForestClassifier 

clfr = RandomForestClassifier(n_estimators = 30) 
start_time = time.time() 
clfr.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time) 

start_time = time.time() 
y_test_pred = clfr.predict(X_train) 
end_time = time.time() 
print("Testing time: ", end_time-start_time) 

print("Train score for RandomForest is:", clfr.score(X_train, y_train)) 
print("Test score for RandomForest is:", clfr.score(X_test, y_test)) 

# Python implementation of Support Vector Classifier
 
from sklearn.svm import SVC 

clfs = SVC(gamma = 'scale')
start_time = time.time()
clfs.fit(X_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ", end_time-start_time)

start_time = time.time()
y_test_pred = clfs.predict(X_train)
end_time = time.time()
print("Testing time: ", end_time-start_time)

print("Train score for VectorClassifier is:", clfs.score(X_train, y_train)) 
print("Test score for VectorClassifier is:", clfs.score(X_test, y_test)) 

# Python implementation of Logistic Regression
