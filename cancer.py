import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


"""Open the dataset using pandas"""
cancer_dataset = pd.read_csv("cancer_dataset.csv")


"""View the first two rows of the data"""
print(cancer_dataset.head(2),"\n")


"""View the names of the columns in the dataset"""
print(cancer_dataset.columns,"\n")


"""In order to create a machine learning model for this current objective a "data" and "target" is needed so in this case the "diagnosis" column
    becomes our target and the rest of the dataset becomes our data"""

data = cancer_dataset.drop("diagnosis",axis=1)

target = cancer_dataset['diagnosis']



"""At this point in order to train our model we need to split our dataset and this is done by using a portion of the data and target as the
Train-set and the rest as the Test-set.
Scikit-Learn has a function (train_test_split) that does this easily and enables us to create train and test values for both x and y axis from the dataset"""

xtrain,xtest,ytrain,ytest = train_test_split(data,target,test_size=.45)


"""Then we initialise our selected machine learning model (in this case the decision trees/random forest model)"""
model = RandomForestClassifier(n_estimators=150,random_state=1)


"""Fit the train sets to the model"""
model.fit(xtrain,ytrain)


"""Use the model to predict possible diagnosis for the xtest parameters"""
predictions = model.predict(xtest)

"""Using accuracy_score, check the accuracy of the prediction with the actual values in ytest"""
score = accuracy_score(predictions,ytest)

print("Accuracy Score =", score * 100)

























































