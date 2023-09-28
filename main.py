import pandas as pd  # data processing, CSV file I/O

#Name            : Sandeep Ramakrishnan
#Student Number  : 202103599
#Stfx Email      : x2021dsw@stfx.ca

# Reading train and test data
train_data = pd.read_csv("C:/Users/admin/PycharmProjects/Machine/train.csv")
print(train_data.head())

test_data_org = pd.read_csv("C:/Users/admin/PycharmProjects/Machine/test.csv")
test_data = pd.read_csv("C:/Users/admin/PycharmProjects/Machine/test.csv")
print(test_data.head())

# Data Preprocessing

# Dropping the values from data set
train_data = train_data.drop(["Name", "PassengerId", "Ticket", "Cabin"], axis=1)
test_data = test_data.drop(["Name", "PassengerId", "Ticket", "Cabin"], axis=1)

# Filling the missing values
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())

train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())

train_gender = train_data["Sex"]
train_gender_list = []

for i in range(len(train_gender)):
    if train_gender[i] == "male":
        train_gender_list.append(1)
    else:
        train_gender_list.append(2)
train_data["Sex"] = train_gender_list

test_gender = test_data["Sex"]
test_gender_list = []

for i in range(len(test_gender)):
    if test_gender[i] == "male":
        test_gender_list.append(1)
    else:
        test_gender_list.append(2)
print(test_gender)
test_data["Sex"] = test_gender_list

train_city = train_data["Embarked"]
print(train_city)
for i in range(len(train_city)):
    if train_city[i] == "S":
        train_city[i] = 1
    elif train_city[i] == "C":
        train_city[i] = 2
    elif train_city[i] == "Q":
        train_city[i] = 3
    else:
        train_city[i] = 1
train_data["Embarked"] = train_city

test_city = test_data["Embarked"]
test_city_list = []
print(test_city)
for i in range(len(test_city)):
    if test_city[i] == "S":
        test_city_list.append(1)
    elif test_city[i] == "C":
        test_city_list.append(2)
    elif test_city[i] == "Q":
        test_city_list.append(3)
    else:
        test_city[i] = 1
test_data["Embarked"] = test_city_list

# Separating label from the train data set
label = train_data["Survived"]
train_data = train_data.drop(["Survived"], axis=1)

# Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_data, label)

# Prediction
predictions = model.predict(test_data)
print(predictions)

# Saving the result
pd.DataFrame({"Survived": predictions, "PassengerId":test_data_org["PassengerId"]}).to_csv("submission.csv", index=False)