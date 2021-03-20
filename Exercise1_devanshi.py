# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:11:57 2021

@author: Devanshi
"""

import pandas as pd
# set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

import os
path = "D:/project_devanshi"
filename = 'titanic.csv'
fullpath = os.path.join(path,filename)
titanic_devanshi = pd.read_csv(fullpath,sep=',')


print(titanic_devanshi.head(3))     #Diplay first 3 records
print(titanic_devanshi.shape)       #Display shape of dataframe


info = pd.DataFrame.info(titanic_devanshi, null_counts=(True)) #Show names, types and counts of columns. 

print(titanic_devanshi.isnull().sum())  #show missing(null values) in columns.

print(titanic_devanshi['Sex'].unique()) #Display Unique value in Sex column
print(titanic_devanshi['Pclass'].unique()) #Display Unique value in pclass column



import matplotlib.pyplot as plt
pd.crosstab(titanic_devanshi.Pclass,titanic_devanshi.Survived).plot(kind='bar')
plt.title('Survivied passengers with respect to their Passenger class')
plt.xlabel('Passanger Class')
plt.ylabel('Number of Passangers Survived')


pd.crosstab(titanic_devanshi.Sex,titanic_devanshi.Survived).plot(kind='bar')
plt.title('Survivied passengers with respect to their Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passangers Survived')

##Scatter Matrix 
temp_columns= ['Survived','Gender','Pclass','Fare','SibSp','Parch'] #6 columns selected as per requirement
to_keep =[i for i in titanic_devanshi if i in temp_columns]

titanic_devanshi_update = titanic_devanshi[to_keep]
pd.plotting.scatter_matrix(titanic_devanshi_update)



##Delete columns which are not suitable for predicting as feature

import numpy as np
drop_columns = ['Cabin','Name','Ticket','PassengerId']
titanic_devanshi_temp = titanic_devanshi.columns.values.tolist()

to_keep = [i for i in titanic_devanshi_temp if i not in drop_columns]

titanic_devanshi_final = titanic_devanshi[to_keep]  #Created new titanic_devanshi which does not include dropped columns



cat_vars= ['Sex','Embarked']    #Selected categorical columns

for var in cat_vars:            #whith the use of get_dummies function, changed values from alphbates to numeric
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(titanic_devanshi[var], prefix=var)
    titanic_devanshi_demo= titanic_devanshi_final.join(cat_list)
    titanic_devanshi_final = titanic_devanshi_demo
    
    
titanic_devanshi_final = titanic_devanshi_final.drop(columns =['Sex','Embarked'])   #Removed all alphbatic columns
 
#placed missing data in Age with mean of the Age 
titanic_devanshi_final['Age'] = titanic_devanshi_final['Age'].replace(np.nan, titanic_devanshi_final['Age'].mean())  

#changed datatype of columns into float - for every row
titanic_devanshi_final = titanic_devanshi_final.astype(float) 

    
#function to normalize all features in dataframe
def normalize(frame_d):
    for col in frame_d:
       
        #x_norm = x - x_min/ x_max- x_min
        
        x_min = frame_d[col].min()
        x_max = frame_d[col].max()
        temp = []
        
        for x in frame_d[col]:
            #frame_d.loc(x)
            #print(x)
            
            x_norm = (x - x_min) / (x_max- x_min)
            
            temp.append(x_norm)
        
        rows = frame_d.shape[0]
        
        for row in range(rows):
            frame_d[col][row] = temp[row]

        
        
normalize(titanic_devanshi_final)    #Called function

print(titanic_devanshi_final.head(2))   #printed 2 records of dataframe



titanic_devanshi_final['Survived'].hist()
plt.title('Histogram of Survived')
plt.xlabel('Survived')
plt.ylabel('Frequency')
titanic_devanshi['Age'].hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')  
titanic_devanshi_final['Pclass'].hist()
plt.title('Histogram of Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Frequency')
titanic_devanshi_final['Fare'].hist()
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
titanic_devanshi_final['SibSp'].hist()
plt.title('Histogram of SibSp')
plt.xlabel('Number of siblings/spouses aboard')
plt.ylabel('Frequency')
titanic_devanshi_final['Parch'].hist()
plt.title('Histogram of Parch')
plt.xlabel('Number of parents/children aboard')
plt.ylabel('Frequency')
titanic_devanshi['Embarked'].hist()
plt.title('Histogram of Embarked')
plt.xlabel('Port of embarkation')
plt.ylabel('Frequency')



#Seperated features into 2 different parts
cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X_devanshi = titanic_devanshi_final[cols]
Y_devanshi = titanic_devanshi_final['Survived']

from sklearn.model_selection import train_test_split

#Sperating featues in ratio of 70-30 percent between train and test
np.random.seed(69)  #Last 2 digit of student no.
X_train_devanshi, X_test_devanshi, Y_train_devanshi, Y_test_devanshi= train_test_split(X_devanshi, Y_devanshi, test_size=0.3, random_state=0)


from sklearn import linear_model
from sklearn import metrics
devanshi_model = linear_model.LogisticRegression(solver='lbfgs')
devanshi_model.fit(X_train_devanshi, Y_train_devanshi)

pd.DataFrame(zip(X_train_devanshi.columns, np.transpose(devanshi_model.coef_)))


#Cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X_train_devanshi, Y_train_devanshi, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

for i in np.arange(0.1,0.55,0.05):
    X_train_devanshi, X_test_devanshi, Y_train_devanshi, Y_test_devanshi= train_test_split(X_devanshi, Y_devanshi, test_size= i, random_state=0)
   
    devanshi_model.fit(X_train_devanshi, Y_train_devanshi)
    scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X_train_devanshi, Y_train_devanshi, scoring='accuracy', cv=10)
    print(round(i*100,0),'% = ',scores.min(),' ',scores.mean(),' ',scores.max())



X_train_devanshi, X_test_devanshi, Y_train_devanshi, Y_test_devanshi= train_test_split(X_devanshi, Y_devanshi, test_size=0.3, random_state=0)

#Calculated probability of Survival whether it will be successful 
y_pred_devanshi = devanshi_model.predict_proba(X_test_devanshi)
print(y_pred_devanshi)

#set threshould value to 50%. Got True False for success and failure
y_pred_devanshi=y_pred_devanshi[:,1]
y_pred_devanshi_flag=pd.DataFrame(y_pred_devanshi)
y_pred_devanshi_flag['predict']=np.where(y_pred_devanshi_flag[0]>=0.5,True,False)



from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

#Accuracy
y_pred_devanshi_flag['predict']= np.where(y_pred_devanshi_flag[0]>=0.5,1,0)

#Confusion Matrix
confusion_m = confusion_matrix(Y_test_devanshi, y_pred_devanshi_flag['predict'])
print(confusion_m)

#Classification_report
classf =  classification_report(Y_test_devanshi, y_pred_devanshi_flag['predict'])
print(classf)

#Changed Threshold value to 50% to 75%

y_pred_devanshi_flag['predict']=np.where(y_pred_devanshi_flag[0]>=0.75,True,False)

#Accuracy for 75 percent threshold
y_pred_devanshi_flag['predict']= np.where(y_pred_devanshi_flag[0]>=0.75,1,0)
acc =  metrics.accuracy_score(Y_test_devanshi, y_pred_devanshi_flag['predict'])
print(acc)

#Confusion Matrix-75% threshold

confusion_m = confusion_matrix(Y_test_devanshi, y_pred_devanshi_flag['predict'])
print(confusion_m)

#classification_report for 75% threshold
classf =  classification_report(Y_test_devanshi, y_pred_devanshi_flag['predict'])
print(classf)





