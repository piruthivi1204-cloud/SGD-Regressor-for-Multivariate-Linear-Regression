# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by: Piruthiviraj G

RegisterNumber:  212225040299


```python
#SGD-ex4
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
# Features (inputs)
X = data[['Size', 'Bedrooms']]

# Targets (outputs)
y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)


price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))

```

## Output:



<img width="324" height="83" alt="image" src="https://github.com/user-attachments/assets/12277f87-086f-4099-bba1-b0d1cfdd6e7c" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
