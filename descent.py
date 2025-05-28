import numpy as np
import matplotlib.pyplot as plt
from flask import Blueprint
from data import area,price,area1

descent=Blueprint("descent", __name__ )

learning_rate = 0.00000001
n_iterations = 77010
n = len(area)
m = 0  
b = 0 
dm = 0
db = 0
for i in range(n_iterations):
       price_pred = m * area + b
       error = price_pred - price
    
       # for j in range(len(price)):
       dm += (-2/n) * sum(area * (price - price_pred))    
       db += (-2/n) * sum(price - price_pred) 

    
       m =m - learning_rate * dm
       b =b - learning_rate * db

    
     
MSE = ((price-price_pred)**2).mean()
print(f"Mean Squared  Error:{MSE}")

RMSE=np.sqrt(MSE)
print(f"Root Mean Squared Error:{RMSE}")

pred_price= (m*area1)+b

print(f"predicted price:{np.round(pred_price,2)}")
print(f"\nLearned slope (m): {m}")
print(f"Learned intercept (b): {b}")


@descent.route("/descent")
def home():
    return f"predicted price:{np.round(pred_price,2)}"

@descent.route("/descent/plot")    
def plotting():
    plt.scatter(area, price, color='blue', label='Actual data')
    plt.plot(area1, pred_price, color='red', label='Predicted data')
    plt.legend()
    plt.xlabel('area')
    plt.ylabel('price')
    plt.title('Simple Linear Regression manual implementation')
    plt.grid(True)
    plt.show()