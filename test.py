import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.metrics import accuracy_score



diabetes_dataset = pd.read_csv('data\diabetes.csv') 

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#preprocessing the data set, replacing null values with the average in (Glucose, BloodPressure, SkinThickness, Insulin, BMI)

imp = SimpleImputer(missing_values=0, strategy='mean')
pregnancy_column = X["Pregnancies"]
temp= X
imp.fit(temp)
temp= X
X = imp.transform(temp)
z = pd.DataFrame(data=X)
z[0] = pregnancy_column
X = z

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.001, random_state=40)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_train)
X_train_prediction = model.predict(X_train)
accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
accuracy = accuracy_score(X_test_prediction, Y_test)



x = [[3, 126, 88, 41, 235, 39.8, 0.704, 27],[1, 117, 88, 24, 145, 34.5, 0.403, 40]]


def predict(x):
    data = pd.DataFrame(data=x)
    standardized_data2 = scaler.transform(x)
    
    return model.predict(standardized_data2)
    

print(predict(x))


"""
x = {"Pregnancies":[3, 1], "Glucose":[126, 117], "BloodPressure":[88, 88],	"SkinThickness":[41, 24],	"Insulin":[235, 34.5], 
"BMI":[39.8, 34.5], "DiabetesPedigreeFunction":[0.704, 0.403], "Age":[27, 40]}
data = pd.DataFrame(data=x)
print(data)
X_test_prediction2 = model.predict(data)
print(X_test_prediction2)
"""

'''
diabetes_dataset = pd.read_csv('data\diabetes.csv') 
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,  random_state=50)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

nonzero=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in nonzero:
    diabetes_dataset[col]=diabetes_dataset[col].replace(0,np.NaN)
    mean=int(diabetes_dataset[col].mean(skipna=True))
    diabetes_dataset[col]=diabetes_dataset[col].replace(np.NaN,mean)

x=diabetes_dataset.iloc[:,0:8]
y=diabetes_dataset.iloc[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.01)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
classifier=KNeighborsClassifier(n_neighbors=15,p=2,metric='euclidean')
model=classifier.fit(x_train,y_train)
yp=classifier.predict(x_test)
CM=confusion_matrix(y_test,yp)
print("F-Score: ",(f1_score(y_test,yp)))
print("Model Accuracy: ",accuracy_score(y_test,yp)*100,"%")
x = {"Pregnancies":[3, 1], "Glucose":[126, 117], "BloodPressure":[88, 88],	"SkinThickness":[41, 24],	"Insulin":[235, 34.5], 
"BMI":[39.8, 34.5], "DiabetesPedigreeFunction":[0.704, 0.403], "Age":[27, 40]}
y = {"Pregnancies":[1], "Glucose":[117], "BloodPressure":[88],	"SkinThickness":[24],	"Insulin":[145], "BMI":[34.5], "DiabetesPedigreeFunction":[0.403], "Age":[40]}
data = pd.DataFrame(data=x)
print(data)
X_test_prediction2 = classifier.predict(data)
print(X_test_prediction2)
'''

"""
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_train)
X_train_prediction = model.predict(X_train)
accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
accuracy_score(X_test_prediction, Y_test)
print(accuracy_score(X_test_prediction, Y_test))

x = {"Pregnancies":[1], "Glucose":117, "BloodPressure":88,	"SkinThickness":24,	"Insulin":145, "BMI":34.5, "DiabetesPedigreeFunction":0.403, "Age":40}
data = pd.DataFrame(data=x)
X_test_prediction2 = model.predict(data)
print(X_test_prediction2)
"""