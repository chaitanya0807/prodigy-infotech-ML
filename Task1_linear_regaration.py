import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the downloaded CSV file
df = pd.read_csv('train.csv')

# Define features and target variable (use relevant columns)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Living area, Bedrooms, Bathrooms
y = df['SalePrice']  # House price

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print("Mean Squared Error:", mse)
print("R-Squared Score:", r2)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predict for a new house example
new_house = np.array([[2000, 3, 2]])  # Example: 2000 sqft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print("Predicted price for new house:", predicted_price)
