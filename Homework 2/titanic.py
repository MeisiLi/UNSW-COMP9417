from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

"""READING DATASET AND CREATING MODEL"""
df = pd.read_csv("titanic.csv")
inputs = df.drop("Survived",axis='columns')
target = df.Survived
normalised_inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.3,random_state=0)
treeModel = tree.DecisionTreeClassifier(random_state=0)
treeModel.fit(x_train,y_train)

#MODEL ACCURACY FOR TESTING AND TRAINING SETS
print("Model accuracy (Testing Set) = ",treeModel.score(x_test,y_test))
print("Model accuracy (Training Set) = ",treeModel.score(x_train,y_train))

#ROC_AUC SCORE FOR TESTING AND TRAINING SETS
tree_roc_auc_test = roc_auc_score(y_test,treeModel.predict(x_test))
tree_roc_auc_train = roc_auc_score(y_train,treeModel.predict(x_train))
print("ROC_AUC SCORE (Testing Set) = ",tree_roc_auc_test)
print("ROC_AUC SCORE (Training Set) = ",tree_roc_auc_train)


#GRID SEARCH ALGO FOR TUNING HYPERPARAMETERS AND SEARCHING OPTIMAL MIN_SAMPLES_LEAF IN TESTING AND TRAINING SETS
param_grid = dict(min_samples_leaf=range(2,21))

gridTraining = GridSearchCV(treeModel,param_grid,scoring = 'roc_auc')
gridTraining.fit(x_train,y_train)

gridTesting = GridSearchCV(treeModel,param_grid,scoring = 'roc_auc')
gridTesting.fit(x_test,y_test)

print("HYPERPARAMETER SEARCH TUNING WITH TRAINING SET :")
print("Optimal score = ",gridTraining.best_score_)
print("Grid Best Params = ",gridTraining.best_params_)

#print("Grid Best Estimator = ",grid.best_estimator_)
#print("CV Results = ",grid.cv_results_)

print()

print("HYPERPARAMETER SEARCH TUNING WITH TESTING SET :")
print("Optimal score = ",gridTesting.best_score_)
print("Grid Best Params = ",gridTesting.best_params_)

plt.plot(range(2,21),gridTraining.cv_results_['mean_test_score'],label='TUNING with Training Set')
plt.plot(range(2,21),gridTesting.cv_results_['mean_test_score'],label='TUNING with Testing Set')
plt.legend()
plt.show()


""" PREDICTING POSTERIOR PROBABILITY """
inputSet_for_prediction = []
for i in range(len(inputs['Pclass'])):
    if inputs['Pclass'][i]==1 and inputs['Sex'][i] == 1:
        inputSet_for_prediction.append([inputs['Pclass'][i],inputs['Sex'][i],inputs['Age'][i],inputs['Siblings_Spouses_Aboard'][i],inputs['Parents_Children_Aboard'][i]])


probs = treeModel.predict_proba(inputSet_for_prediction)[:,1]

print("PROBABILITY OF A WOMAN FIRST CLASS PASSENGER TO SURVIVE IS ",(sum(probs)/len(probs))*100)
