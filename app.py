from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))  # Make sure model.pkl is in the same folder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age'))
        anaemia = float(request.form.get('anaemia'))
        cpk = float(request.form.get('creatinine_phosphokinase'))
        diabetes = float(request.form.get('diabetes'))
        ef = float(request.form.get('ejection_fraction'))
        hbp = float(request.form.get('high_blood_pressure'))
        platelets = float(request.form.get('platelets'))
        sc = float(request.form.get('serum_creatinine'))
        ss = float(request.form.get('serum_sodium'))
        sex = float(request.form.get('sex'))
        smoking = float(request.form.get('smoking'))
        time = float(request.form.get('time'))

        features = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time]])
        result = model.predict(features)

        output = "Patient is **likely to die**" if result[0] == 1 else "Patient is **likely to survive**"
        return render_template('index.html', result=output)

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
