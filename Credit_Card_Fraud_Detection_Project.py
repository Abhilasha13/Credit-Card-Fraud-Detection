#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# It is really important to detect any Credit Card Fraud taking place. The credit card companies need to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

# In this kernel, I have worked on the Credit Card Fraud Detection dataset. The dataset was taken from Kaggle.com.
# Here, I have performed exploratory data analysis, undersampling and predictive modelling.

# <b> Importing the libraries


# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA, TruncatedSVD
#import matplotlib.patches as mpatches
#import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# Reading the dataset in credit_data
credit_data = pd.read_csv('/Users/ashutoshshanker/Downloads/creditcard.csv')

credit_data.head()


# Columns in the dataset
credit_data.columns


# <b> Data Preprocessing


# Finding the count and percentage of missing values
missing_data = pd.DataFrame([credit_data.isnull().sum(), credit_data.isnull().sum()* 100.0/credit_data.shape[0]]).T
missing_data.columns = ['missing_count', 'missing_percentage']
missing_data


# Summary Statistics of the dataset
credit_data.describe().T


# Count of fraudulent and non-fraudulent transactions
credit_data["Class"].value_counts()


# There are 492 fraud cases and 284315 non-fraud cases. This shows that the dataset is unbalanced and so, for the purpose of prediction, we need to have a equal distribution of both fraud and non-fraud cases.

# Plot showing fraudulent and non-fraudulent transactions 
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
sns.FacetGrid(credit_data, hue="Class", size = 6).map(plt.scatter, "Time", "Amount", edgecolor="k").add_legend()
plt.show()


# # Correlation between the columns

# Correlation between the columns
credit_data.corr()

# heatmap to find any high correlations

plt.figure(figsize=(10,10))
sns.heatmap(data=credit_data.corr(), cmap="seismic")
plt.show();


# # Pearson's Correlation

plt.figure(figsize=(8,2))
credit_data.corr()['Class'].sort_values()[:-1].plot(kind='bar')
plt.show()


# # Distribution
# Here, we will find the distribution to understand the skewness of data in amount and time column

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = credit_data['Amount'].values
time_val = credit_data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize = 14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color = 'b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])


# Record Counts
credit_data.count()


# <b> Histogram Plot

# Histogram Plot
credit_data.hist(figsize=(20, 20));


# <b> Boxplot

data_box=credit_data.drop('Class',axis=1)


# distribution of the features and target
cols= list(data_box.columns)
for i in cols:
    sns.boxplot(y=i,x=credit_data['Class'],data=data_box)
    plt.show()


# <b> Since most of the columns has already been scaled we should scale the columns that are left to scale i.e. Amount and Time columns

# Scaling the columns using StandardScaler, RobustScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers. So, here we will use RobustScaler.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

credit_data['scaled_amount'] = rob_scaler.fit_transform(credit_data['Amount'].values.reshape(-1,1))
credit_data['scaled_time'] = rob_scaler.fit_transform(credit_data['Time'].values.reshape(-1,1))

credit_data.drop(['Time','Amount'], axis=1, inplace=True)


# # Distribution of fradulent and non-fraudulent data

# Percentage of fradulent and non-fraudulent data
pd.DataFrame(credit_data["Class"].value_counts(), credit_data["Class"].value_counts()*100/len(credit_data["Class"]))


# Bar plot showing the class ditribution before undersampling
colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=credit_data, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# # Undersampling the data:
# 
# To balance the data undersampling is performed.
# 
# Since, there are 492 fraud cases and 284315 non-fraud cases, the dataset is unbalanced and so, for the purpose of prediction, we need to have a equal distribution of both fraud and non-fraud cases. 
# 
# Note: The main issue with "Random Under-Sampling" is that we run the risk that our classification models will not perform as accurate as we would like to since there is a great deal of information loss 

X = credit_data.ix[:, credit_data.columns != 'Class']
y = credit_data.ix[:, credit_data.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(credit_data[credit_data.Class == 1])
fraud_indices = np.array(credit_data[credit_data.Class == 1].index)

# Picking the indices of the non-fraudulent classes
normal_indices = credit_data[credit_data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = credit_data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# Bar plot showing the class ditribution after undersampling
colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=under_sample_data, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# # Data Prediction

# Dividing the dataset into training and testing data:

X = credit_data.drop('Class', axis = 1)
Y = credit_data.Class


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)

# Weight of transaction
#wp = y_train.value_counts()[0] / len(y_train)
#wn = y_train.value_counts()[1] / len(y_train)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {X_train.shape}")
print(f"y_test: {y_test.shape}")
print(f"X_train_undersample: {X_train_undersample.shape}")
print(f"X_test_undersample: {X_test_undersample.shape}")
print(f"y_train_undersample: {y_train_undersample.shape}")
print(f"y_test_undersample: {y_test_undersample.shape}")


# # Logistic Regression

# <b> Logistic Regression on original data

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# <b> Logistic Regression on undersampled data

logreg = LogisticRegression()
logreg.fit(X_train_undersample, y_train_undersample)


# Accuracy of model
y_pred_undersample = logreg.predict(X_test_undersample)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_undersample, y_test_undersample)))


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
print(confusion_matrix)


# Calculating the precision, recall, f1-score, and support
from sklearn.metrics import classification_report
print(classification_report(y_test_undersample, y_pred_undersample))


# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_undersample, logreg.predict(X_test_undersample))
fpr, tpr, thresholds = roc_curve(y_test_undersample, logreg.predict_proba(X_test_undersample)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Decision Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()


# Decision tree model on original dataset
classifier.fit(X_train, y_train)

y_pred_dt = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_dt))

# Calculating the precision, recall, f1-score, and support
print(classification_report(y_test, y_pred_dt))


# <b> Decision Tree on undersampled data

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_undersample, y_train_undersample)


y_pred_undersample_dt = classifier.predict(X_test_undersample)

from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test_undersample, y_pred_undersample_dt))

# Calculating the precision, recall, f1-score, and support
print(classification_report(y_test_undersample, y_pred_undersample_dt))


# # Naive Bayes

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model = GaussianNB()

# Naive Bayes on original Data
model.fit(X_train, y_train)

y_pred_NB = model.predict(X_test)
y_pred_NB

accuracy = accuracy_score(y_test,y_pred_NB)*100
accuracy


# Naive Bayes on undersampled Data
model.fit(X_train_undersample, y_train_undersample)

y_pred_undersample_NB = model.predict(X_test_undersample)
y_pred_undersample_NB


accuracy_undersample = accuracy_score(y_test_undersample,y_pred_undersample_NB)*100
accuracy_undersample


# # SVM


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel



#SVM on original dataset takes long time to run hence, I have commented this code:
# SVM on original dataset
#Train the model using the training sets
#clf.fit(X_train, y_train)

#Predict the response for test dataset
#y_pred_svm = clf.predict(X_test)




#Accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
#Precision_svm = metrics.precision_score(y_test, y_pred)
#Recall_svm = metrics.recall_score(y_test, y_pred)


# SVM on undersampled dataset
#Train the model using the training sets
clf.fit(X_train_undersample, y_train_undersample)

#Predict the response for test dataset
y_pred_undersample_svm = clf.predict(X_test_undersample)


# Calculating Accuracy, Precision, and Recall
Accuracy_svm = metrics.accuracy_score(y_test_undersample, y_pred_undersample_svm)
Precision_svm = metrics.precision_score(y_test_undersample, y_pred_undersample_svm)
Recall_svm = metrics.recall_score(y_test_undersample, y_pred_undersample_svm)

# Accuracy
Accuracy_svm

# Precision
Precision_svm

# Recall
Recall_svm


# # Neural Network

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# Neural Network on original dataset
clf.fit(X_train, y_train)


y_pred_NN = clf.predict(X_test)

# Accuracy
Accuracy_NN = metrics.accuracy_score(y_test, y_pred_NN)
Accuracy_NN

# Neural Network on undersampled data
clf.fit(X_train_undersample, y_train_undersample)

y_pred_undersample_NN = clf.predict(X_test_undersample)

# Accuracy
Accuracy_undersample_NN = metrics.accuracy_score(y_test_undersample, y_pred_undersample_NN)
Accuracy_undersample_NN


# It can be observed that the accuracy is more for original data in each case. This is caused due to the unbalanced data in the 'Class' column. 

# # Performing Feature Selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


feat_labels = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
       'scaled_amount', 'scaled_time']


# For undersample
# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train_undersample, y_train_undersample)

# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf, threshold=0.15)

# Train the selector
sfm.fit(X_train_undersample, y_train_undersample)


# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])


# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train_undersample = sfm.transform(X_train_undersample)
X_important_test_undersample = sfm.transform(X_test_undersample)


# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train_undersample, y_train_undersample)


# Apply The Full Featured Classifier To The Test Data
y_pred_undersample = clf.predict(X_test_undersample)

# View The Accuracy Of Our Full Feature (4 Features) Model
accuracy_score(y_test_undersample, y_pred_undersample)


# Apply The Full Featured Classifier To The Test Data
y_important_pred_undersample = clf_important.predict(X_important_test_undersample)

# View The Accuracy Of Our Limited Feature (2 Features) Model
accuracy_score(y_test_undersample, y_important_pred_undersample)


# conda install -c conda-forge imbalanced-learn


# # References:
# 
# 1. https://www.kaggle.com/mlg-ulb/creditcardfraud
# 2. https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
