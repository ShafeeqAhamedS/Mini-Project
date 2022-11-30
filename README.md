# Liver Cirrhosis Analysis & Prediction
## About Liver Cirrhosis
* Chronic liver damage from a variety of causes leading to scarring and liver failure.
* Hepatitis and chronic alcohol abuse are frequent causes.
* Liver damage caused by cirrhosis can't be undone, but further damage can be limited.
* Initially patients may experience fatigue, weakness and weight loss.
* During later stages, patients may develop jaundice (yellowing of the skin), gastrointestinal bleeding, abdominal swelling and confusion.
## About the dataset
* This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. 
* The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). 
* This data set contains 441 male patient records and 142 female patient records.

* Any patient whose age exceeded 89 is listed as being of age "90".
## Program
## Importing Modules

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
```

## Data Pre-Processing

```py
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 _Intro to DS/Mini_Project/Liver.csv")
df
df.head()
df.info()
df.describe()
df.tail()
df.shape
df.columns

df.duplicated()
df.duplicated().sum()
df[df.duplicated()]
df=df.drop_duplicates()
df.duplicated().sum()

df.isnull().sum()
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())
df.isnull().sum()

df['Dataset']=df['Dataset'].map({1:1,2:0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df
```
### Dataset
![image](https://user-images.githubusercontent.com/93427237/204739741-06eb1b1b-450e-4dbe-81a7-fe2bd542c4b5.png)
## Detecing & Removing Outliers

```py
df.drop('Dataset',axis=1).plot(kind='box',layout=(2,5),subplots=True,figsize=(12,6))
plt.show()

q1=df['Total_Bilirubin'].quantile(0.25)
q3=df['Total_Bilirubin'].quantile(0.75)
iqr=q3-q1
df['Total_Bilirubin']=df['Total_Bilirubin'].apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x).apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x)

q1=df['Direct_Bilirubin'].quantile(0.25)
q3=df['Direct_Bilirubin'].quantile(0.75)
iqr=q3-q1
df['Direct_Bilirubin']=df['Direct_Bilirubin'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)

q1=df['Alkaline_Phosphotase'].quantile(0.25)
q3=df['Alkaline_Phosphotase'].quantile(0.75)
iqr=q3-q1
df['Alkaline_Phosphotase']=df['Alkaline_Phosphotase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)


q1=df['Alamine_Aminotransferase'].quantile(0.25)
q3=df['Alamine_Aminotransferase'].quantile(0.75)
iqr=q3-q1
df['Alamine_Aminotransferase']=df['Alamine_Aminotransferase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)


q1=df['Aspartate_Aminotransferase'].quantile(0.25)
q3=df['Aspartate_Aminotransferase'].quantile(0.75)
iqr=q3-q1
df['Aspartate_Aminotransferase']=df['Aspartate_Aminotransferase'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)


q1=df['Total_Protiens'].quantile(0.25)
q3=df['Total_Protiens'].quantile(0.75)
iqr=q3-q1
df['Total_Protiens']=df['Total_Protiens'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)


q1=df['Albumin_and_Globulin_Ratio'].quantile(0.25)
q3=df['Albumin_and_Globulin_Ratio'].quantile(0.75)
iqr=q3-q1
df['Albumin_and_Globulin_Ratio']=df['Albumin_and_Globulin_Ratio'].apply(lambda x : q3+(1.5*iqr) if x>q3+(1.5*iqr) else x).apply(lambda x : q1-(1.5*iqr) if x<q1-(1.5*iqr) else x)

df.drop('Dataset',axis=1).plot(kind='box',layout=(2,5),subplots=True,figsize=(12,6))
plt.show()
```
### Before removing outliers
<img width="493" alt="image" src="https://user-images.githubusercontent.com/93427237/204740023-a1607e51-2771-4838-8a50-e9f34f0d3407.png">

### After removing outliers
<img width="493" alt="image" src="https://user-images.githubusercontent.com/93427237/204740103-0128a899-a6df-49c8-9817-b6f3618c06ce.png">

## Heatmap

```py
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap='magma',annot=True)
plt.show()
```
<img width="550" alt="image" src="https://user-images.githubusercontent.com/93427237/204740272-87bb84fc-567b-4e0b-9d26-c0d1fc4355a5.png">

## Correlation

```py
plt.figure(figsize=(10,5))
df.corr()['Dataset'].sort_values(ascending=False).plot(kind='bar',color='black')
plt.xticks(rotation=90)
plt.xlabel('Variables in the Data')
plt.ylabel('Correlation Values')
plt.show()
```
<img width="550" alt="image" src="https://user-images.githubusercontent.com/93427237/204740363-7091f51b-d917-450c-8741-948add1d2767.png">

## EDA

* We have 10 independent variables to predict the class (y-variable).
* The class contains binary values either 0 or 1. 
* Notation 0 stands for non-liver patients and 1 stands for people with liver disease.
* We have data of 416 people with liver problem and 167 people without liver problem

```py
df['Dataset'].value_counts()
```
<img width="400" alt="image" src="https://user-images.githubusercontent.com/93427237/204748924-4f227205-5049-4b93-b98e-9f6efa9cb7ec.png">


```py
asc = df["Age"].value_counts().sort_values(ascending=False).index

plt.figure(figsize = (18,6))
sns.countplot(data=df,x=df["Age"],hue = df["Dataset"],order=asc)
plt.title('Count across Age',fontsize=16)
plt.show()
```
<img width="788" alt="image" src="https://user-images.githubusercontent.com/93427237/204749440-141c2ed5-d82f-4185-bea1-785e44295ec4.png">

```py
plt.figure(figsize=(9,6))
df['Gender'].value_counts().plot(kind='pie',autopct='%.2f%%',colors=['goldenrod','dimgray'],explode=(0,0.2),shadow=True)
label=['Male','Female']
plt.title('Percentage difference in the count between Male and Female')
plt.legend(label,title='Category')
plt.show()
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/93427237/204749181-1d45eda6-4912-4e8d-b091-80e346839365.png">

```py
plt.figure(figsize=(9,6))
df['Dataset'].value_counts().plot(kind='pie',autopct='%.2f%%',colors=['deepskyblue','lawngreen'],explode=(0,0.2),shadow=True)
plt.title('Percentage difference in the count between Healthy & Diseased')
plt.show()
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/93427237/204749274-ead7a190-56c4-4fba-9a57-212fd547781a.png">

```py
sns.countplot(df['Dataset'],palette='Set2')
plt.title('Count Across Disease')
plt.show()
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/93427237/204740688-0b8072bb-a5b5-4737-a082-605060d79d2a.png">


```py
sns.countplot(x="Dataset", hue="Gender", data=df)
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/93427237/204749720-4049347a-965f-4329-a0f1-525fc95cc111.png">

```py
plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(8,6))
plt.margins(x=0)
plt.title('Histogram for Age with Disease')
sns.histplot(x = df["Age"], hue = df["Dataset"], palette="winter_r", kde=True)
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204750148-e15bd2c0-e7c4-4c1b-b0f6-65554447b116.png">

1. People starts to get the Liver Disease from the age of 25 to 50 
2. There is chances because of the youngster consuming lot of junk food and Processed foods. 
3. May be also there is possibilty No taking proper food at proper time

## Relationship

### Direct_Bilirubin vs Total_Bilirubin

```py
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204740800-7fa859e0-090b-4fa9-99ff-63438a1c9092.png">

There seems to be direct relationship between Total_Bilirubin and Direct_Bilirubin.

### Aspartate_Aminotransferase vs Alamine_Aminotransferase

```py
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204740826-c3ae0cc5-edf9-48b2-95a5-c25b07414195.png">

There is linear relationship between Aspartate_Aminotransferase and Alamine_Aminotransferase and the gender.

### Alkaline_Phosphotase vs Alamine_Aminotransferase

```py
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=df, kind="reg")
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204740858-450124f4-765a-4ebb-b3cf-d7ae3ecc8e72.png">

No linear correlation between Alkaline_Phosphotase and Alamine_Aminotransferase

### Total_Protiens vs Albumin

```py
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204740882-6e6421d6-e13f-4972-b40e-39b9efb55daf.png">
There is linear relationship between Total_Protiens and Albumin and the gender.

### Albumin vs Albumin_and_Globulin_Ratio

```py
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")
```
<img width="450" alt="image" src="https://user-images.githubusercontent.com/93427237/204740899-7454340d-1a65-4690-8341-d3c690ddd0a2.png">

There is linear relationship between Albumin_and_Globulin_Ratio and Albumin.

From the above jointplots and scatterplots, we find direct relationship between the following features:
1. Direct_Bilirubin & Total_Bilirubin
2. Aspartate_Aminotransferase & Alamine_Aminotransferase
3. Total_Protiens & Albumin
4. Albumin_and_Globulin_Ratio & Albumin

## Applying ML
### Assinging X and Y
```py
X = df.drop(['Dataset'], axis=1)
y = df['Dataset']
```
### Normalization
```py
scaler=MinMaxScaler()
scaled_values=scaler.fit_transform(X)
X.loc[:,:]=scaled_values
```
#### Before Normalization
![norm](./befrnorm.png)
#### After Normalization
![norm](./aftrnorm.png)
### Splitting Data 
```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)
print("Training sample shape =",X_train.shape)
print("Testing sample sample =",X_test.shape)
```
![split](./split.png)
### Applying Logisitc Regression
```py
reg = LogisticRegression()
reg.fit(X_train, y_train)

log_predicted= reg.predict(X_test)
### Measuring accuracy
```py
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")
print('Classification Report: \n', classification_report(y_test,log_predicted))
```
#### Accuray
![acc](./acc.png)
#### Confusion Matrix
![conf](./conf.png)</br>
<img width="350" alt="image" src="./conf1.png">
#### Classification Matrix
![class](./report.png)

## Predicitng
```py
pred = reg.predict([[22,0,50,10.5,100,120,50,5.0,2.5,1.2]])
if(pred == 1):
  print("Infected with Liver Cirrohisis")
else:
  print("Not Infected with Liver Cirrohisis")
```
![pred](./pred.png)
