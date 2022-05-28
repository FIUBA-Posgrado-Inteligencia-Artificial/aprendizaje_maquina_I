
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('diabetes_api')

# Define predict function
@app.post('/predict')
def predict(Number_of_times_pregnant, Plasma_glucose_concentration_a_2_hours_in_an_oral_glucose_tolerance_test, Diastolic_blood_pressure_(mm_Hg), Triceps_skin_fold_thickness_(mm), 2_Hour_serum_insulin_(mu_U/ml), Body_mass_index_(weight_in_kg/(height_in_m)^2), Diabetes_pedigree_function, Age_(years)):
    data = pd.DataFrame([[Number_of_times_pregnant, Plasma_glucose_concentration_a_2_hours_in_an_oral_glucose_tolerance_test, Diastolic_blood_pressure_(mm_Hg), Triceps_skin_fold_thickness_(mm), 2_Hour_serum_insulin_(mu_U/ml), Body_mass_index_(weight_in_kg/(height_in_m)^2), Diabetes_pedigree_function, Age_(years)]])
    data.columns = ['Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)', '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'Diabetes pedigree function', 'Age (years)']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)