
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/wineq"
data = pd.read_csv(url, sep=';')
data

# Load the wine quality dataset
#data = pd.read_csv('winequality-white.csv', delimiter=';')
# Split the dataset into training and testing sets
X = data[['alcohol']].values
y = data['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0


# Fit a linear regression model to the data
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)
# Fit a polynomial regression model to the data
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
                                                    
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)
# Predict the quality of the wine for the test data using both models
y_pred_lin = lin_regressor.predict(X_test)
y_pred_poly = poly_regressor.predict(X_poly_test)
# Print the performance metrics for both models
print('Linear Regression Metrics:')
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print('Mean Squared Error: ', mse_lin)
print('Root Mean Squared Error: ', rmse_lin)
print('R-squared: ', r2_lin)

                                                    

print('Polynomial Regression Metrics:')
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print('Mean Squared Error: ', mse_poly)
print('Root Mean Squared Error: ', rmse_poly)
print('R-squared: ', r2_poly)
                                                    

# Plot the learning curves for both models
train_sizes, train_scores_lin, test_scores_lin = learning_curve(lin_regressor, X, y
train_sizes, train_scores_poly, test_scores_poly = learning_curve(poly_regressor, X
train_mean_lin = np.mean(train_scores_lin, axis=1)
train_std_lin = np.std(train_scores_lin, axis=1)
test_mean_lin = np.mean(test_scores_lin, axis=1)
test_std_lin = np.std(test_scores_lin, axis=1)
train_mean_poly = np.mean(train_scores_poly, axis=1)
train_std_poly = np.std(train_scores_poly, axis=1)
test_mean_poly = np.mean(test_scores_poly, axis=1)
test_std_poly = np.std(test_scores_poly, axis=1)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(train_sizes, train_mean_lin, label='Training score')
plt.plot(train_sizes, test_mean_lin, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean_lin - train_std_lin, train_mean_lin + trai
plt.fill_between(train_sizes, test_mean_lin - test_std_lin, test_mean_lin + test_st
plt.xlabel('Training set size')
plt.show()
                 

import matplotlib.pyplot as plt
# Plot the learning curves for Linear Regression
plt.plot(train_sizes, train_mean_lin, label='Training Score (Linear Regression)')
plt.plot(train_sizes, train_mean_poly, label='Validation Score (Linear Regression)
# Plot the learning curves for Polynomial Regression
plt.plot(train_sizes, test_mean_lin, label='Training Score (Polynomial Regression)
plt.plot(train_sizes, test_mean_poly, label='Validation Score (Polynomial Regressio
# Set the plot title and labels
plt.title("Learning Curves for Linear and Polynomial Regression")
plt.xlabel("Training examples")
plt.ylabel("Score")
# Set the legend
plt.legend(loc="best")
# Show the plot
plt.show()
         
