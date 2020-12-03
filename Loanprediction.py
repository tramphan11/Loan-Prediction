#In this project i will use all three classification algorithms which are 
#1. Logistic Regression 
#2. Random Tree
#3. Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df_train_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#print(df_train.head()

#now we will check for missing values in the dataset using ISNULL 
total = df_train.isnull().sum().sort_values(ascending=False) 
#print(total.head(20))

#fill in the missing values 
df_train['LoanAmount'] = df_train['LoanAmount'].fillna(df_train['LoanAmount'].mean())
df_train['Credit_History'] = df_train['Credit_History'].fillna(df_train['Credit_History'].median())

#print(df_train.isnull().sum().sort_values(ascending=False))
print(df_train.isnull().sum().sort_values(ascending=False))

#exploratory data analysis with the help of graphs and figures 
plt = sns.countplot(df_train['Gender'],hue=df_train['Loan_Status'])
print(pd.crosstab(df_train['Gender'],df_train['Loan_Status']))

plt2= sns.countplot(df_train['Married'], hue=df_train['Loan_Status'])
print(pd.crosstab(df_train['Married'],df_train['Loan_Status']))

plt3= sns.countplot(df_train['Education'], hue=df_train['Loan_Status'])
print(pd.crosstab(df_train['Education'],df_train['Loan_Status']))

#replacing the variables in the datasaet with numerical values 

df_train['Loan_Status'].replace('Y',1,inplace=True)
df_train['Loan_Status'].replace('N',0,inplace=True)
df_train['Loan_Status'].value_counts()

df_train.Gender=df_train.Gender.map({'Male':1,'Female':0})
df_train['Gender'].value_counts()

df_train.Married=df_train.Married.map({'Yes':1,'No':0})
df_train['Married'].value_counts()

df_train.Dependents=df_train.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df_train['Dependents'].value_counts()

df_train.Education=df_train.Education.map({'Graduate':1,'Not Graduate':0})
df_train['Education'].value_counts()

df_train.Self_Employed=df_train.Self_Employed.map({'Yes':1,'No':0})
df_train['Self_Employed'].value_counts()

df_train.Property_Area=df_train.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df_train['Property_Area'].value_counts()

df_train['LoanAmount'].value_counts()

'''
#desplaying the correlation mix 
#plt.figure.Figure(figsize=(16,5)
sns.heatmap(df_train.corr(), annot=True)
plt.title('Correlation Matrix (for Loan Status)')
'''

#now importing the packages for the classification of data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#using logistic regression 
model = LogisticRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
evaluation = f1_score(y_test, ypred)
# print('Logistic Regression accuracy = ', metrics.accuracy_score(y_pred ,y_test))

#using decision tree 
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
print(y_pred_tree)

evaluation_acc = f1_score(y_test , y_pred_tree)
print(evaluation_acc)

#using random forest classifier 
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

y_pred_forest = forest.predict(X_test)
print(y_pred_forest)

eval_acc = f1_score(y_test, y_pred_forest)
print(eval_acc)

