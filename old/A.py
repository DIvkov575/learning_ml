import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load Training and Test Data
train_raw_data = pd.read_csv('train.csv')
test_raw_data = pd.read_csv('test.csv')
y_raw_test = pd.read_csv('gender_submission.csv')


# Function for preprocessing
def preprocessing(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    data.loc[data['Age'].isnull(), ['Age']] = data['Age'].median()
    data.loc[data['Fare'].isnull(), ['Fare']] = data['Age'].mean()
    data = pd.get_dummies(data)

    try:
        data.pop('Survived')
    except:
        pass

    X_columns = data.columns
    normalize = StandardScaler()
    X = normalize.fit_transform(data)
    X = pd.DataFrame(X, columns=X_columns)
    return X


# Train Model
def train(train_raw_data):
    X_train = preprocessing(train_raw_data)
    y_train = train_raw_data["Survived"]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# Initilaize Training and Test Data
X_train = preprocessing(train_raw_data)
