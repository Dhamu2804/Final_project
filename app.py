from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and process crop data
crop_data = pd.read_csv('Crop_recommendation.csv')
crop_features = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_target = crop_data['label']
Xtrain_crop, Xtest_crop, Ytrain_crop, Ytest_crop = train_test_split(crop_features, crop_target, test_size=0.2, random_state=2)
crop_rf = RandomForestClassifier(n_estimators=20, random_state=0)
crop_rf.fit(Xtrain_crop, Ytrain_crop)

# Load and process fertilizer data
fertilizer_data = pd.read_csv('Fertilizer Prediction.csv')
fertilizer_data.columns = fertilizer_data.columns.str.strip()
fertilizer_data_encoded = pd.get_dummies(fertilizer_data, columns=['Crop Type', 'Soil Type'])
y_fertilizer = fertilizer_data_encoded['Fertilizer Name']
X_fertilizer = fertilizer_data_encoded.drop('Fertilizer Name', axis=1)
Xtrain_fertilizer, Xtest_fertilizer, Ytrain_fertilizer, Ytest_fertilizer = train_test_split(
    X_fertilizer, y_fertilizer, train_size=0.7, shuffle=True, random_state=1
)
fertilizer_rf = RandomForestClassifier(n_estimators=50, random_state=0)
fertilizer_rf.fit(Xtrain_fertilizer, Ytrain_fertilizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            inputs = {key: float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']}
            input_data = np.array([list(inputs.values())])
            crop_prediction = crop_rf.predict(input_data)[0]
            return render_template('crop_result.html', prediction=crop_prediction)
        except ValueError:
            return render_template('crop_predict.html', error="Invalid input. Please enter valid numbers.")
    return render_template('crop_predict.html')

@app.route('/predict_fertilizer', methods=['GET', 'POST'])
def predict_fertilizer():
    if request.method == 'POST':
        data = request.form
        crop_type = data['crop_type']
        soil_type = data['soil_type']
        try:
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            moisture = float(data['moisture'])
            nitrogen = float(data['nitrogen'])
            potassium = float(data['potassium'])
            phosphorous = float(data['phosphorous'])
        except ValueError:
            return render_template('fertilizer_predict.html', error="Invalid input. Please enter valid numbers.")

        input_data = pd.DataFrame([{
            'Crop Type': crop_type,
            'Soil Type': soil_type,
            'Temperature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorous
        }])
        input_data_encoded = pd.get_dummies(input_data)
        input_data_encoded = input_data_encoded.reindex(columns=X_fertilizer.columns, fill_value=0)

        fertilizer_prediction = fertilizer_rf.predict(input_data_encoded)[0]
        return render_template('fertilizer_result.html', result=fertilizer_prediction)
    return render_template('fertilizer_predict.html')

if __name__ == '__main__':
    app.run(debug=True)
