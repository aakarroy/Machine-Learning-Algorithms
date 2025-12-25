# importing all libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import matplotlib.pyplot as plt

# reading data
raw_data = pd.read_csv(r"framingham.csv")

# filing data
for i in ["cigsPerDay","BPMeds","totChol","heartRate","glucose"]:
    median = raw_data[i].median()
    raw_data[i] = raw_data[i].fillna(int(median))
median = raw_data["BMI"].median()
raw_data["BMI"] = raw_data["BMI"].fillna(median)
mode = raw_data["education"].mode()[0]
raw_data["education"] = raw_data["education"].fillna(mode)

# extracting features and labels from data
features = raw_data[["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]
label = raw_data["TenYearCHD"]
input_train, input_test,target_train, target_test = model_selection.train_test_split(features,label,test_size=0.1)
# print(raw_data.isna().sum()) --> column wise sum of nan values
# print(raw_data.describe())  --> describes data using statistics  
# print(raw_data.corr())  --> draws corr matrix of the data

# describing model 
model = LogisticRegression(solver="liblinear")
model.fit(input_train,target_train) # --> training of model
y_pred = model.predict(input_test)  # --> taking predictions from the model
prob = model.predict_proba(input_test)
p,r,t = metrics.roc_curve(target_test,prob[:,1]) 
f1 = 2*(p*r)/(p+r+1e-7) # --> calculating f1 score
best = np.argmax(f1)
new_pred = (prob[:,1]>=t[best]).astype(int) # --> new prediction based on best threshold
# print(metrics.accuracy_score(target_test,new_pred)) --> accuracy
# print(metrics.confusion_matrix(target_test,new_pred)) --> confusion matrix
# print(f"{t[best]}\n") --> best threshold
auc = metrics.roc_auc_score(target_test, prob[:,1])

# plotting Roc curve
plt.plot(p,r,color = "red",lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier') # --> reference line

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
