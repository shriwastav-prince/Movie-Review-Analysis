import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#loading train data
train_data=pd.read_csv("train.csv")

movie_data=pd.read_csv('movies.csv')

duplicates_mask = movie_data.duplicated(subset=['movieid'], keep='first')

# Drop rows with duplicate values in the 'movieid' column
movie_data.drop_duplicates(subset=['movieid'], keep='first', inplace=True)

# If you want to reset the index after dropping duplicates (optional)
movie_data.reset_index(drop=True, inplace=True)

train=pd.merge(train_data,movie_data,on='movieid',how='left')

#Data Visualisation on sentiment column
#countplot of sentiment using seaborn
sns.set(style='darkgrid')
sns.countplot(data=train, x='sentiment')
sentiment_counts = train['sentiment'].value_counts()
# Set the plot title and labels
# plt.title('Sentiment Count')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')

# # Show the plot
# plt.show()




# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='audienceScore',y='rating',data=train)
# plt.xlabel('Audience Score')
# plt.ylabel('Rating')
# plt.title('Distribution of Audience Score')
# plt.show()



train['reviewText'].fillna('', inplace=True)

# Drop unnecessary columns (you can choose to keep other columns based on your model requirements)
train.drop(columns=["movieid", "reviewerName", "isFrequentReviewer"], inplace=True)

# Drop columns with high missing value percentage
train.drop(columns=["boxOffice", "releaseDateTheaters"], inplace=True)

#Separating feature and target
features=['reviewText','originalLanguage','genre','runtimeMinutes','audienceScore','rating','ratingContents','distributor','director']
X = train[features]
y = train['sentiment']



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# from sklearn.metrics import classification_report,confusion_matrix


# Define column transformer to process different types of features
numeric_features = ['runtimeMinutes','audienceScore']
categorical_features = ['genre','originalLanguage','rating','ratingContents','distributor','director']
text_feature = 'reviewText'

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

from sklearn.feature_extraction.text import TfidfVectorizer

#Creating the instance of Tfidf
tfidf_transformer = TfidfVectorizer()

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', tfidf_transformer, text_feature)
    ])


#applying preprocessing steps on training data
X_preprocessed = preprocessor.fit_transform(X)


#splitting the train dataset for validation (80:20)
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

#Logistic Regression Model
baseline_model = LogisticRegression()

#Hyperparameter tuning
param_grid = {
    'C': [ 0.1, 0.5, 1, 10],  # Regularization parameter
    'penalty': ['l1', 'l2']  #  Regularization type ('l1' = Lasso, 'l2' = Ridge)
}
grid_search = GridSearchCV(baseline_model, param_grid, cv=7)
grid_search.fit(X_train, y_train)

baseline_model = grid_search.best_estimator_

# Model evaluation
y_pred = baseline_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)


from sklearn.metrics import roc_curve, auc, roc_auc_score
# Convert categorical sentiment labels to binary values (1 for POSITIVE, 0 for NEGATIVE)
y_test_binary = np.where(y_test == 'POSITIVE', 1, 0)
y_pred_binary = np.where(y_pred == 'POSITIVE', 1, 0)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)


# Plot the ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# Calculate AUC score using roc_auc_score
auc_score = roc_auc_score(y_test_binary, y_pred_binary)



from sklearn.naive_bayes import MultinomialNB

# Create and train the Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

#Hyperparameter tuning
param_grid={
    'alpha':[0.1,1.0,2.0]
}

grid_search = GridSearchCV(naive_bayes_classifier,param_grid,cv=7)
grid_search.fit(X_train, y_train)
naive_bayes_classifier.fit(X_train, y_train)

naive_bayes_classifier=grid_search.best_estimator_

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)


# Convert categorical sentiment labels to binary values (1 for POSITIVE, 0 for NEGATIVE)
y_test_binary = np.where(y_test == 'POSITIVE', 1, 0)
y_pred_binary = np.where(y_pred == 'POSITIVE', 1, 0)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)

# Calculate AUC score using roc_auc_score
auc_score = roc_auc_score(y_test_binary, y_pred_binary)


# Create a dictionary for the data
data = {
    'Model': ['Baseline', 'Naive Bayes'],
    'Accuracy': [0.80, 0.78],
    'Precision (Negative)': [0.72, 0.72],
    'Precision (Positive)': [0.84, 0.79],
    'Recall (Negative)': [0.65, 0.52],
    'Recall (Positive)': [0.87, 0.90],
    'F1-score (Negative)': [0.68, 0.60],
    'F1-score (Positive)': [0.85, 0.84],
    'Support (Negative)': [10696, 10696],
    'Support (Positive)': [21856, 21856],
    'AUC Score': [0.76, 0.71]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)



test=pd.read_csv('test.csv')
test_data=pd.merge(test,movie_data,on='movieid',how='left')

test_data['reviewText'].fillna('', inplace=True)

test_data=test_data[features]

#applying feature extraction on test data using pipeline
test1=preprocessor.transform(test_data)


new_reviews_pred = baseline_model.predict(test1)

new_reviews_pred=pd.Series(new_reviews_pred)
new_reviews_pred.value_counts()


# Creating DataFrame of predicted output as required for submission
submission = pd.DataFrame(columns=['id', 'sentiment'])
submission['id'] = [i for i in range(len(new_reviews_pred))]
submission['sentiment'] = new_reviews_pred

# # Converting the DataFrame to CSV file for submision
submission.to_csv("submission.csv", index=False)




import pickle

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)
