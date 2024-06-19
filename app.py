from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

app = Flask(__name__)

# Define the data
years = list(range(1950, 2021))
renewable_energy = [4250, 4492, 4733, 4975, 5217, 5458, 5700, 5942, 6183, 6425, 6667, 6908, 7150, 7392, 7633, 7875, 8117, 8358, 8600, 8842, 9083, 9425, 9767, 10108, 10450, 10792, 11133, 11475, 11817, 12158, 12500, 12942, 13383, 13825, 14267, 14708, 15150, 15592, 16033, 16475, 16917, 17917, 18917, 19917, 20917, 21917, 22917, 23917, 24917, 25917, 26917, 28917, 30917, 32917, 34917, 36917, 38917, 40917, 42917, 44917, 46917, 51917, 56917, 61917, 66917, 71917, 76917, 81917, 86917, 91917, 96917]

data = {'Year': years, 'Renewable Energy (MW)': renewable_energy}
data = pd.DataFrame(data)

# Split the data into training and test sets
train_data = data[data['Year'] <= 2015]
test_data = data[data['Year'] > 2015]

# Fit the SARIMA model
model_sarima = SARIMAX(train_data['Renewable Energy (MW)'], order=(5, 1, 0), seasonal_order=(1, 0, 1, 12))
model_sarima_fit = model_sarima.fit(disp=0)

# Save the model to a pickle file
pickle_file_path = 'model_sarima_fit.pkl'
with open(pickle_file_path, 'wb') as f:
    pickle.dump(model_sarima_fit, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    year = None
    if request.method == 'POST':
        year = int(request.form['year'])
        with open(pickle_file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Make prediction
        start_year = 1950
        end_year = 2015
        
        if year <= end_year:
            # Prediction within the training period
            prediction = data.loc[data['Year'] == year, 'Renewable Energy (MW)'].values[0]
        else:
            # Prediction beyond the training period
            num_years_to_predict = year - end_year
            predictions = loaded_model.predict(start=len(train_data), end=len(train_data) + num_years_to_predict - 1, dynamic=True)
            prediction = predictions.iloc[-1]
        prediction = round(prediction, 2)    

    return render_template('index.html', prediction=prediction, year=year)

if __name__ == '__main__':
    app.run(debug=True)
