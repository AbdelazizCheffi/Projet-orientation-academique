from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib

# regrouper Col_domaine_activite_pro1 ,filière

app = Flask(__name__)

# Load the Random Forest model and encoders
model_rf, encoder, label_encoder = joblib.load('RandomForest.pkl')

# Define ordinal columns (make sure it matches your training data)
ordinal_columns = ['mention', 'type_etu', 'tranche_age', 'Niveau_etu']

@app.route('/')
def index():
    return render_template('index.html')


def predict():
    # Get user inputs from the form
    user_inputs = {
    # #  'mention': request.form['mention'],
    # #   'type_etu': request.form['type_etu'],
    # #   "tranche_age" : request.form['tranche_age'],
    # #   "Niveau_etu" : request.form['Niveau_etu'],
    # #   "bac" : request.form['bac'],
        "filière" : request.form['filière'],
    # #   "type_lieu_etu" : request.form['type_lieu_etu'],
    # #   "genre" : request.form['genre'],
    # #   "satisfact_formation" : request.form['satisfact_formation'],
    # #  "Col_specialite1" : request.form['Col_specialite1'],
    #    "Col_matière_prefere1" : request.form['Col_matière_prefere1'],
    #    "Col_matière_prefere2" : request.form['Col_matière_prefere2'],
        "Col_domaine_activite_pro1" : request.form['Col_domaine_activite_pro1'], 
    # #   "domaine_etu" : request.form['domaine_etu'],     
    }

 # Create a DataFrame from user inputs
    user_data = pd.DataFrame([user_inputs])

    # Encode ordinal columns
    user_data[ordinal_columns] = encoder.transform(user_data[ordinal_columns])

    # Encode the target variable
    user_data['domaine_etu'] = label_encoder.transform(user_data['domaine_etu'])

    # Make predictions using the loaded model
    predictions = model_rf.predict(user_data)

    # Return the predictions as JSON
    return jsonify({'prediction': predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)    


