# importing libraries
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# getting data
raw_data = pd.read_csv(r"mashhad_apartment_sales_2024.csv")

# converting data
raw_data["unit_per_floor"] = pd.to_numeric(raw_data["unit_per_floor"],errors="coerce")
raw_data["total_floors_count"] = pd.to_numeric(raw_data["total_floors_count"],errors="coerce")
print("So corr of unit_per_floor, unit_per_floor, construction_year is very low and is unwanted we remove it")
# droping data
raw_data.drop(["unit_per_floor","total_floors_count","construction_year","created_at_shamsi","created_at_date","Price(Toman)"],axis=1,inplace=True)

# filling data
print("filing nan values in rooms_count with the median value")
raw_data["rooms_count"] = raw_data["rooms_count"].fillna(raw_data["rooms_count"].median())

# one hot encoding
enc = preprocessing.OneHotEncoder()
enc.fit(raw_data[["neighborhood"]])
one_hot = enc.transform(raw_data[["neighborhood"]]).toarray()
cols = enc.get_feature_names_out(["neighborhood"])
one_hot_df = pd.DataFrame(one_hot,columns=cols)
raw_data = pd.concat([raw_data,one_hot_df],axis=1)
raw_data.drop(["neighborhood"],axis=1,inplace=True)

features = raw_data.copy()
features.drop(["Price(USD)"],axis=1,inplace=True)
labels = raw_data["Price(USD)"]

# spliting dataset
input_train, input_test, target_train, target_test = model_selection.train_test_split(features,labels,test_size=0.1,random_state=100,shuffle=True)

# defining model
model = LinearRegression()
model.fit(input_train,target_train)
predictions = model.predict(input_train) # --> getting predictions 
plt.scatter(predictions,target_train)
plt.plot([min(predictions),max(predictions)],[min(predictions),max(predictions)],"--g") # --> ploting predictions
plt.title("Prediction vs Actual values")
plt.show()
print(model.coef_) # --> slope of line
print(model.intercept_) # --> intercept of line
rmse = metrics.root_mean_squared_error(target_train,predictions)
print(rmse) # --> rmse loss

