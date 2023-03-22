# import necessary modules
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# ignore convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# load digits dataset
dataset=load_digits()

# split data into features and target
X=dataset.data
y=dataset.target

# print feature and target names
print("Features: ", dataset.feature_names)
print("Targets: ", dataset.target_names)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=30)

# define list of models to evaluate
L=[svm.SVC(),LogisticRegression(max_iter=1000),KNeighborsClassifier(n_neighbors=5)]
accuracy_list=[]

# evaluate each model and store accuracy score in list
i=0
for model in L:
    tempo_list=['SVM','Logistique','Kneighbors']
    model.fit(X_train, y_train)

    test_prediction=model.predict(X_test)
    train_prediction=model.predict(X_train)

    train_accuracy=accuracy_score(y_train,train_prediction)
    test_accuracy =accuracy_score(y_test,test_prediction)

    accuracy_list.append(str(tempo_list[i])+" : "+str(test_accuracy))
    i+=1

# print accuracy scores for each model
print(accuracy_list)

