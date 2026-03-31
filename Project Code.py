import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("car_fuel_efficiency_dataset.csv")

X = data[["engine_size", "weight", "horsepower", "cylinders"]]
y = data["mpg"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Example Prediction
example = [[2.0, 1200, 100, 4]]
prediction = model.predict(example)

print("Predicted MPG for sample input:", prediction[0])
