import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("StudentsPerformance.xls")
st.title("Selamat datng di aplikasi test murid")
check_data = st.checkbox("Lihat contoh data")
if check_data:
    st.write(data.head())
#data.head() #
st.write("Mari kita lihat hasil test nya")
#data.head() #
check_nan = data['math score'].isnull().values.any() #check any nan on math score
print (check_nan)
count_nan = data['math score'].isnull().sum() #count any nan on math score
print (count_nan)
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
Gender = LabelEncoder()
data['gender'] = Gender.fit_transform(data['gender'])
data.head()
Race = LabelEncoder()
data['race/ethnicity'] = Race.fit_transform(data['race/ethnicity'])
Parental = LabelEncoder()
data['parental level of education'] = Parental.fit_transform(data['parental level of education'])
lunch = LabelEncoder()
data['lunch'] = lunch.fit_transform(data['lunch'])
TestPrep = LabelEncoder()
data['test preparation course'] = TestPrep.fit_transform(data['test preparation course']) #change testprep
bins = (-1,40,50,60,80,100)
study = ['Skip Class', 'Sleeping in Class' , 'Npc' , 'Good Student' , 'Lord']
data['math score'] = pd.cut(data['math score'], bins = bins, labels = study)
data.head()
sns.relplot(x='reading score',y="writing score", hue="math score", data= data) #just to make it cool
#but its actually functional since we know that person who good at reading and writing are good at math
check_nan = data['math score'].isnull().values.any()
print (check_nan)
count_nan = data['math score'].isnull().sum() #check nan again in case there is some data in math score who doesnt get category format
print (count_nan)
X = data.drop('math score',axis=1)
y = data['math score'] #math score
X_train, X_test, y_train, y_test = train_test_split(X.values,y, test_size = 0.5, random_state = 0) #split the dataset
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
clf = svm.SVC()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred)) #accuracy
a = [[0,3,0,1,1,60,90]] #women,group d,associate's degree, standard lunch, none preparation, 60 reading score, 90 writing score
a = s.transform(a)
b = clf.predict(a)
b

