import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## Membaca dataset
data = pd.read_csv("StudentsPerformance.xls")
st.title("Selamat datang di aplikasi test murid")
## Menampilkan data
check_data = st.checkbox("Tampilkan data")
if check_data:
    st.write(data.head())
#data.head() #
## Memasukan input untuk diprediksi
sex = st.radio("Masukan jenis kelamin",('female','male'))
## Mengklasifikasi jenis kelamin menjadi angka agar bisa diproses
if sex=="female":
    sex=0
else :
    sex=1
etnik= st.selectbox( "Masukan Group Etnis ",('group A','group B','group C','group D'))
## Mengklasifikasi etnik menjadi angka agar bisa diproses
if etnik == "group A":
    etnik=0
elif etnik =="group B":
    etnik=1
elif etnik =="group C":
    etnik=2
else :
    etnik=3
pend_ortu= st.selectbox( "Masukan Pendidikan terakhir orang tua ",('some high','some school','some college','high school',"master's degree","associate's degree"))
## Mengklasifikasi pendidikan orangtua menjadi angka agar bisa diproses
if pend_ortu == "some high":
    pend_ortu=0
elif pend_ortu =="some school":
    pend_ortu=1
elif pend_ortu =="some college":
    pend_ortu=2
elif pend_ortu =="high school":
    pend_ortu=3
elif pend_ortu =="master's degree":
    pend_ortu=4
else :
    pend_ortu=5
maksi= st.selectbox( "Masukan Jenis Makan siang ",('standard','Free/reduced','none'))
## Mengklasifikasi makan siang menjadi angka agar bisa diproses
if maksi == "standard":
    maksi=0
elif maksi =="free/reduced":
    maksi=1
else :
    maksi=2
#kursus = st.number_input('Masukan nilai Kursus (dari 0-100) :')
kursus = st.radio("Apakah pernah mengikuti kursus",('completed','none'))
## Mengklasifikasi kursus menjadi angka agar bisa diproses
if kursus=="completed":
    kursus=0
else :
    kursus=1
baca = st.number_input('Masukan nilai Membaca (dari 0-100) :')
tulis = st.number_input('Masukan nilai Menulis (dari 0-100) :')
st.write("Mari kita lihat hasil test nya")
#data.head() #
## Cek apakah ada nan di math score
check_nan = data['math score'].isnull().values.any() #check any nan on math score
#print (check_nan)
check_data = st.checkbox("Check nan")
if check_data:
    st.write(check_nan)
## menghitung jumlah nan
count_nan = data['math score'].isnull().sum() #count any nan on math score
#print (count_nan)
check_data = st.checkbox("Check jumlah nan dari math score")
if check_data:
    st.write(count_nan)
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse',False)
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
st.pyplot()
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
#checking prediction house price
if st.button("Cek Prediksi?"):
    print(classification_report(y_test,pred))
    print(accuracy_score(y_test,pred)) #accuracy
    a = [[sex,etnik,pend_ortu,maksi,kursus,baca,tulis]] #women,group d,associate's degree, standard lunch, none preparation, 60 reading score, 90 writing score
    a = s.transform(a)
    b = clf.predict(a)
    b
