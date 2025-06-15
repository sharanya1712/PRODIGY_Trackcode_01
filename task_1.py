import numpy as np
data = [
    [1500, 3, 2, 250000],
    [2000, 4, 3, 350000],
    [1200, 2, 1, 180000],
    [1800, 3, 2, 300000],
    [2500, 4, 3, 420000],
    [1300, 2, 2, 200000],
    [1700, 3, 1, 280000],
    [2200, 4, 2, 380000],
    [1000, 2, 1, 150000],
    [1900, 3, 2, 320000],
]
data_array = np.array(data)
X_raw = data_array[:, :-1]
y = data_array[:, -1]

X = np.c_[np.ones(X_raw.shape[0]), X_raw]
lambda_reg = 1e-3
I = np.eye(X.shape[1])
I[0, 0] = 0
Xt_X = X.T @ X
inv_Xt_X = np.linalg.inv(Xt_X + lambda_reg * I)
theta = inv_Xt_X @ X.T @ y
print("\nRegression Coefficients:")
print(f"Intercept: {theta[0]:,.2f}")
print(f"SqFt: {theta[1]:,.2f}, Bedrooms: {theta[2]:,.2f}, Bathrooms: {theta[3]:,.2f}")
def predict(features):
    features = np.insert(features, 0, 1) 
    return features @ theta
new_houses = np.array([[1600, 3, 2], [2100, 4, 3], [900, 1, 1]])
for house in new_houses:
    price = predict(house)
    print(f"\nPredicted price for house {house}: ${price:,.2f}")
y_pred_train = X @ theta
mae = np.mean(np.abs(y - y_pred_train))
mse = np.mean((y - y_pred_train) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - (np.sum((y - y_pred_train) ** 2) / np.sum((y - np.mean(y)) ** 2))

print("\nModel Evaluation Metrics:")
print(f"MAE: ${mae:,.2f}, MSE: ${mse:,.2f}, RMSE: ${rmse:,.2f}, R²: {r2:.4f}")