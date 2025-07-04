import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

def lift(x,n_poly=3):
    N = np.size(x, 0)
    X = np.zeros([N,n_poly+1])
    for p in range(n_poly+1):
        X[:,p]=x**p
    return X

def fit(X, y, ridge=False, l=None):
    return np.linalg.inv(X.T @ X + l * np.eye(X.shape[1])) @ X.T @ y if ridge else np.linalg.inv(X.T @ X) @ X.T @ y 

data1 = np.loadtxt('data/nonlindata1.csv', delimiter=',')
data2 = np.loadtxt('data/nonlindata2.csv', delimiter=',')
data3 = np.loadtxt('data/nonlindata3.csv', delimiter=',')

x_test=np.arange(-2,2,.01)

plt.style.use('ggplot')
for i, data in enumerate([data1, data2, data3]):
    x, y = data[:, 0], data[:, 1]
    X = lift(x, 12)
    X_test = lift(x_test, 12)

    #Linear Least Square Regression
    t = fit(X, y)
    y_hat = X_test @ t
    
    plt.scatter(x, y, edgecolors="black")
    plt.plot(x_test, y_hat, color="blue")
    plt.title(f"Linear Least Squares Regression on data{i + 1}")
    plt.show()

    #Ridge Regression
    ts = [(l, fit(X, y, True, l)) for l in [0.0001, 0.01, 1, 100, 10000]]
    y_hats = [(l, X_test @ t) for l,t in ts]

    for l, y_hat in y_hats:
        plt.figure()
        plt.scatter(x, y, edgecolors="black")
        plt.plot(x_test, y_hat, color='blue')
        plt.title(f"Ridge Regression on data{i + 1}, λ = {l}")
        plt.legend()
        plt.show()

    #Huber Regression
    huber = HuberRegressor().fit(X, y)
    y_hat = huber.predict(X_test)

    plt.figure()
    plt.scatter(x, y, edgecolors="black")
    plt.plot(x_test, y_hat, color="blue")
    plt.title(f"Huber Regression on data{i + 1}")
    plt.show()