import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
X = data.drop(columns=["price"])

# Assuming "Unnamed: 0" is present in the original data
if "Unnamed: 0" in X.columns:
    X = X.drop(columns=["Unnamed: 0"])

y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use make_column_transformer without specifying remainder
column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ["location"]))

scaler = StandardScaler(with_mean=False)
ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(X_train, y_train)

# Save the model without the need to save the column transformer and scaler separately
pickle.dump(pipe, open("RidgeModel.pkl", "wb"))

p = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data["location"].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    sqft = request.form.get("total_sqft")
    print(location, bhk, bath, sqft)

    input_data = pd.DataFrame({"location": [location], "total_sqft": [sqft], "bath": [bath], "bhk": [bhk]})
    print(input_data)

    # Make predictions
    prediction = p.predict(input_data)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
