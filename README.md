# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME  : YOGESH RAO S D
REG NO: 212222110055
```
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/5f546fbc-aa3d-42f3-a343-b38cbc22d65f)
```
data.isnull().sum()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/702a9d71-d409-4016-8d62-97f14cd78804)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/4d4fec15-4169-4240-97cc-c998d39be6ff)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/c66b9b34-7cfe-4702-9b47-e2a2665e9876)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/f7830929-1d59-401b-aace-5ac2e3f05810)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/e2a03295-80f4-4473-9b1b-eec90764b21e)
```
data2
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/291744cd-47f2-498f-a151-094570f4c230)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/22a237a7-1d58-4fd4-8ac9-eea6b28feaa7)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/4bf3c71f-6aa8-476f-ba37-775e9e719cb9)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/43329ad4-a007-454d-bf59-da13d02525f4)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/3e104c36-f36a-486a-82bf-0d7fa190f105)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/16b8ecf7-abd4-4c84-b41d-9dba0401913b)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/637909f6-abf8-4001-b907-1fc3238eaafc)
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/02c01c89-cf8e-49cf-91c4-2f82958a56fd)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/e8b52cf0-4a8d-4f37-b482-f8474169fd98)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/e5e4a190-ec19-4e27-a4be-3f5fd83424b7)
```
data.shape
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/ab402668-0ed6-4bcc-8efe-88a30234a0d3)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/07124b6b-5fcc-4be4-84c9-8ff7f460fa5b)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/4e2e4948-68bd-4032-8993-9bcb353c0683)
```
tips.time.unique()
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/dcf0947a-794a-4a7f-9fa3-eb90ac8d0d65)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/b1ada218-79d9-4f0f-8f9c-dff2f83dab42)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/bharathganeshsivasankaran/EXNO-4-DS/assets/119478098/3371ef7a-8779-41f7-9e13-154f4de94f63)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
