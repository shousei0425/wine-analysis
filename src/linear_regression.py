import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/wine.csv")


X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values


X_b = np.hstack([np.ones_like(X), X])


beta, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)


intercept = beta[0]
slope = beta[1]

print("切片:", intercept)
print("回帰係数（alcohol）:", slope)


y_pred = intercept + slope * X


plt.scatter(X, y, alpha=0.5, label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression (Least Squares Method)")
plt.legend()

plt.savefig("regression_result.png")
plt.close()
