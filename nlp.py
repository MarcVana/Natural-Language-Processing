"""
Created on Wed Oct  7 18:20:17 2020

NLP PROJECT BASED ON YELP DATASET (from Kaggle.com)

@author: Marc
"""
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
yelp = pd.read_csv('yelp.csv')
yelp['text length'] = yelp['text'].apply(len)

# Visualizing and understanding the data
g = sns.FacetGrid(yelp, col = 'stars')
g.map(plt.hist, 'text length', bins = 50)
plt.savefig('stars_vs_textlength.png')
plt.close()

sns.countplot(x = 'stars', data = yelp, palette = 'rainbow')
plt.savefig('stars_count.png')
plt.close()

stars = yelp.groupby('stars').mean()

sns.heatmap(stars.corr(), cmap = 'coolwarm', annot = True)
plt.savefig('correlations.png')
plt.close()

# NLP Task

# Only with 1 or 5 stars reviews
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
X = yelp_class['text']
Y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_test)

# Printing the results
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, predictions))
print('-----------------------------------------')
print(classification_report(Y_test, predictions))

# With pipeline and TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tf-idf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators = 500))
    ])

X = yelp_class['text']
Y = yelp_class['stars']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
pipeline.fit(X_train, Y_train)
predictions = pipeline.predict(X_test)

# Printing the new results
print(confusion_matrix(Y_test, predictions))
print('-----------------------------------------')
print(classification_report(Y_test, predictions))






