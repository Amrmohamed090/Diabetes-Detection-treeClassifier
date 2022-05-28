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

X.loc[X['Glucose'] == 0,'Glucose'] = np.nan
X.loc[X['BloodPressure'] == 0,'BloodPressure'] = np.nan
X.loc[X['SkinThickness'] == 0,'SkinThickness'] = np.nan
X.loc[X['Insulin'] == 0,'Insulin'] = np.nan
X.loc[X['BMI'] == 0,'BMI'] = np.nan
X.loc[X['DiabetesPedigreeFunction'] == 0,'DiabetesPedigreeFunction'] = np.nan
X.loc[X['Age'] == 0,'Age'] = np.nan


#remove outlier
lower_limit = X["Insulin"].quantile(0.1)  
upper_limit = X["Insulin"].quantile(0.90)

X["Insulin"] = np.where(X["Insulin"]> upper_limit, upper_limit,
                        np.where(X["Insulin"]< lower_limit, lower_limit,
                        X["Insulin"]))
''''''
lower_limit = X['BloodPressure'].quantile(0.05)  
upper_limit = X['BloodPressure'].quantile(0.95)

X['BloodPressure'] = np.where(X['BloodPressure']> upper_limit, upper_limit,
                        np.where(X['BloodPressure']< lower_limit, lower_limit,
                        X['BloodPressure']))

''''''
lower_limit = X["SkinThickness"].quantile(0.05)  
upper_limit = X["SkinThickness"].quantile(0.95)

X["SkinThickness"] = np.where(X["SkinThickness"]> upper_limit, upper_limit,
                        np.where(X["SkinThickness"]< lower_limit, lower_limit,
                        X["SkinThickness"]))
''''''
lower_limit = X["BMI"].quantile(0.05)  
upper_limit = X["BMI"].quantile(0.95)

X["BMI"] = np.where(X["BMI"]> upper_limit, upper_limit,
                        np.where(X["BMI"]< lower_limit, lower_limit,
                        X["BMI"]))

''''''
lower_limit = X["Pregnancies"].quantile(0)  
upper_limit = X["Pregnancies"].quantile(0.99)

X["Pregnancies"] = np.where(X["Pregnancies"]> upper_limit, upper_limit,
                        np.where(X["Pregnancies"]< lower_limit, lower_limit,
                        X["Pregnancies"]))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pregnancy_column = X["Pregnancies"]
temp= X
imp.fit(temp)
temp= X
X = imp.transform(temp)
z = pd.DataFrame(data=X)
z[0] = pregnancy_column
X = z
X.head()
X.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, stratify=Y, random_state=2)


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

