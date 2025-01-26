import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('data.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')

# Save the model to a file
import joblib
joblib.dump(model, 'linear_regression_model.joblib')

# Function to make predictions with new data
def make_prediction(new_data):
    # Load the model
    loaded_model = joblib.load('linear_regression_model.joblib')
    # Make a prediction
    prediction = loaded_model.predict(new_data)
    return prediction
