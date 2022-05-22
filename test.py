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


score = 0
best_sample_split = 2
accuracy_list = list()
from sklearn.tree import DecisionTreeClassifier
score = 0
best_sample_leaf = 2
accuracy_list = list()
for i in range(2,100):
    model = DecisionTreeClassifier(random_state=0, min_samples_leaf= i)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_train)
    X_train_prediction = model.predict(X_train)
    accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = model.predict(X_test)
    new_score = accuracy_score(X_test_prediction, Y_test)
    accuracy_list.append(new_score)
    if new_score > score:
        score = new_score
        best_sample_leaf = i
        
    
model = DecisionTreeClassifier(random_state=0, min_samples_leaf=best_sample_leaf )
model.fit(X_train, Y_train)

predictions = model.predict(X_train)
X_train_prediction = model.predict(X_train)
accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
new_score = accuracy_score(X_test_prediction, Y_test)
accuracy_score(X_test_prediction, Y_test)



x = [[3, 126, 88, 41, 235, 39.8, 0.704, 27],[1, 117, 88, 24, 145, 34.5, 0.403, 40]]


def predict(x):
    data = pd.DataFrame(data=x)
    standardized_data2 = scaler.transform(x)
    
    return model.predict(standardized_data2)
    

print(predict(x))

