# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start
#### Step 2: Import Libraries
#### Step 3: Load the Dataset
#### Step 4: Split the Data
#### Step 5: Instantiate the Model
#### Step 6: Train the Model
#### Step 7: Make Predictions
#### Step 8: Evaluate the Model
#### Step 9: Stop

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Visalan H
RegisterNumber: 212223240183
```
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df=pd.read_csv('Placement_Data.csv')
df.isnull().sum()
df['salary'].fillna((df['salary'].mean()),inplace=True)
df.isnull().sum()

X=df.drop(columns=['sl_no','status'])
y=df['status']
le=LabelEncoder()

columns_to_encode = ['gender', 'ssc_b', 'hsc_b', 'degree_t', 'workex', 'specialisation','hsc_s']
for column in columns_to_encode:
    X[column] = le.fit_transform(X[column])

mm=MinMaxScaler()

col_minmax = ['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']
for col in col_minmax:
    X[col] = mm.fit_transform(X[[col]])

y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.27,random_state=38)

lr=LogisticRegression(max_iter=1000,tol=1e-5,solver='lbfgs')
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
acc=accuracy_score(y_test,pred)
print(confusion_matrix(y_test,pred))
print("Visalan H")
print("212223240183")
print(classification_report(y_test,pred))

scaled_pred=[[0,0.084725,1,0.329489,1,1,0.195122,0,1,0.133125,1,0.514993,0.119805]]
print("Visalan H")
print("212223240183")
lr.predict(scaled_pred)

df.iloc[100:101,:]
```

## Output:
### Preprocessing:
![image](https://github.com/user-attachments/assets/64269a93-6aae-4e87-b9d4-f61e7d443e0d)

### Classifictaion: 
![image](https://github.com/user-attachments/assets/73e66ae0-1613-401e-b160-6187257c099d)

### Prediction:
![image](https://github.com/user-attachments/assets/ff1d679a-021f-4841-a855-30388c435f1b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
