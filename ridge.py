import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from flask import Blueprint
from data import area, price, area1

ridge=Blueprint("ridge", __name__,static_folder="static",template_folder="template" )

model =Ridge() 
model.fit(area,price)

predict=model.predict(area)

m=model.coef_[0]
b=model.intercept_

MSE=mean_squared_error(price,predict)
print(f"Mean Squared Error:{MSE}")

RMSE=np.sqrt(MSE)
print(f"Root Mean Squared Error:{RMSE}")

print(f"\nLearned slope (m): {m}")
print(f"Learned intercept (b): {b}")


pred_price=model.predict(area1)



@ridge.route("/ridge")
def  home():
    return f"predicted price:{pred_price}"

@ridge.route("/ridge/plot")    
def plotting():
    plt.scatter(area, price, color='blue', label='Actual data')
    plt.plot(area1, pred_price, color='red', label='Predicted data')
    plt.legend()
    plt.xlabel('area')
    plt.ylabel('price')
    plt.title('Simple Linear Regression using Gradient Descent & Lasso(scikit learn)')
    plt.grid(True)
    plt.show()