import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み（実行場所が wine-analysis の場合）
df = pd.read_csv("data/wine.csv")

# 説明変数（例：alcohol）と目的変数（quality）
X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values

# 切片項（1の列）を追加
X_b = np.hstack([np.ones_like(X), X])

# 最小二乗法で回帰係数を求める
beta, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)

# 回帰係数（切片と傾き）
intercept = beta[0]
slope = beta[1]

print("切片:", intercept)
print("回帰係数（alcohol）:", slope)

# 予測値
y_pred = intercept + slope * X

# 図の作成
plt.scatter(X, y, alpha=0.5, label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression (Least Squares Method)")
plt.legend()

plt.show()
