import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

# Read in the dataset
with open("full_set.txt") as f:
    content = f.readlines()

# Remove leading and trailing white space
content = [x.strip() for x in content]

# Separate the sentences from the labels
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]

# Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
y = np.array(labels, dtype='int8')
y = 2*y - 1

# Full remove function to clean up text
def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x

# Remove digits and punctuation
digits = [str(x) for x in range(10)]
digit_less = [full_remove(x, digits) for x in sentences]
punc_less = [full_remove(x, list(string.punctuation)) for x in digit_less]

# Make everything lower-case
sents_lower = [x.lower() for x in punc_less]

# Define and remove stop words
stop_set = set(['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from'])
sents_split = [x.split() for x in sents_lower]
sents_processed = [" ".join(list(filter(lambda a: a not in stop_set, x))) for x in sents_split]

# Transform the bag of words representation
vectorizer = CountVectorizer(analyzer="word", max_features=4500)
data_features = vectorizer.fit_transform(sents_processed)

# Append '1' to the end of each vector for the bias term
data_mat = np.ones((data_features.shape[0], data_features.shape[1]+1))
data_mat[:,:-1] = data_features.toarray()

# TRAINING/TEST SPLIT
np.random.seed(0)

# Ensure that the test set has an equal number of positive and negative samples
negative_indices = np.where(y == -1)[0]
positive_indices = np.where(y == 1)[0]

test_negative_indices = np.random.choice(negative_indices, 500, replace=False)
test_positive_indices = np.random.choice(positive_indices, 500, replace=False)

test_inds = np.concatenate([test_negative_indices, test_positive_indices])
train_inds = list(set(range(len(labels))) - set(test_inds))

train_data = data_mat[train_inds, :]
train_labels = y[train_inds]
test_data = data_mat[test_inds, :]
test_labels = y[test_inds]

# Fitting a logistic regression model to the training data
clf = SGDClassifier(loss="log_loss", penalty=None, random_state=0)
clf.fit(train_data, train_labels)

# Get predictions on training and test data
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)

# Compute errors
errs_train = np.sum(preds_train != train_labels)
errs_test = np.sum(preds_test != test_labels)

print("Training error: ", float(errs_train)/len(train_labels))
print("Test error: ", float(errs_test)/len(test_labels))
