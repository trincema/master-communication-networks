## Intrusion Detection System
Introduction:
Intrusion Detection System is a software application to detect network intrusion using various machine learning algorithms.IDS monitors a network or system for malicious activity and protects a computer network from unauthorized access from users, including perhaps insider. The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between ‘bad connections’ (intrusion/attacks) and a ‘good (normal) connections’.

Attacks fall into four main categories:
#DOS: denial-of-service, e.g. syn flood;
#R2L: unauthorized access from a remote machine, e.g. guessing password;
#U2R: unauthorized access to local superuser (root) privileges, e.g., various “buffer overflow” attacks;
#probing: surveillance and another probing, e.g., port scanning.

Various Algorithms Applied: Gaussian Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, Logistic Regression.

Approach Used: I have applied various classification algorithms that are mentioned above on the KDD dataset and compare there results to build a predictive model.

## References
- https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/
- https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
