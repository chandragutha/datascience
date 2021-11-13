import os
import pandas as pd

# Working directory
os.chdir("D:/chandrawork/datascience/ml-models-practice/kaggle-ghoost-classification")
ghoost_train = pd.read_csv("train.csv")


# Data Analysis


#Separating the dependent variable from the independent variables
y_train = ghoost_train.iloc[:,-1]
print(y_train)

x_train = ghoost_train.iloc[:,1:-1]
print(x_train)



#For encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
labelencoder_COL= LabelEncoder()
#y_train= labelencoder_Y.fit_transform(y_train)  
#print(y_train)
x_train['color']=labelencoder_COL.fit_transform(x_train['color'])

#print(x_train)


#build the decision tree model on train data
#Checking correlation between the independent variables
#print(ghoost_train.corr()) #not much correlation amongst the variables


#For getting the accuracy of the trained models 
from sklearn import metrics

#TRAINING THE DATA ON DIFFERENT MODELS

#Naive-Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_train)
print(metrics.accuracy_score(y_train,y_pred))
#0.749 accuracy


#predict the outcome
ghoost_test = pd.read_csv("test.csv")
x_test= ghoost_test
x_test = ghoost_test.iloc[:,1:]

print(x_test)

#For encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
labelencoder_COL= LabelEncoder()
#y_train= labelencoder_Y.fit_transform(y_train)  
#print(y_train)
x_test['color']=labelencoder_COL.fit_transform(x_test['color'])
print(x_test)


ghoost_test['type'] = classifier.predict(x_test)
ghoost_test.to_csv("submission_classi.csv", columns=['id','type'], index=False)