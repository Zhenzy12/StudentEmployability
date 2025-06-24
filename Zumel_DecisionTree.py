import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# read the dataset
df = pd.read_csv('Student-Employability-Datasets.csv')

# separate the input features (X) and target variable (y)
x = df.iloc[:, 1:-1].values  # Exclude Name of Student column
y = df.iloc[:, -1].values

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=8)

# create a decision tree classifier
clf = DecisionTreeClassifier()

# fit the classifier to the training data
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# print the classification report
print(classification_report(y_test, y_pred))

# get input values for new student from user
new_student = [5,5,5,5,5,5,5,5]

# new_student = []
# # define column names for new student data
# columns = ['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 'Student Performance Rating']
# for col in columns:
#     value = int(input(f"Enter {col}: "))
#     new_student.append(value)

# make a prediction for the new student using the trained classifier
prediction = clf.predict([new_student])

# print the prediction
print(f"Prediction: {prediction[0]}")

# save the trained model using pickle
with open('student_employability.pkl', 'wb') as f:
    pickle.dump(clf, f)
