import os
import pandas as pd
from sklearn import tree


# Working directory
os.chdir("D:/ChandraWork/DataScience/practicemodels/Titanic-classification")
titanic_train = pd.read_csv("train.csv")

# Data Analysis
titanic_train.shape
titanic_train.info()
titanic_train.groupby('Survived').size()
titanic_train1 = pd.get_dummies(titanic_train, columns = ['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
Y_train = titanic_train['Survived']

#build the decision tree model on train data
dt = tree.DecisionTreeClassifier()
dt.fit(X_train,Y_train)

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()


titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()
titanic_test1.head(6)

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)