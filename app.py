from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image

model = pickle.load(open('model1.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('crop4.html')

@app.route('/predict', methods = ['POST'])
def recomend_crop():
    N = float(request.form.get('N'))
    P = float(request.form.get('P'))
    K = float(request.form.get('K'))
    temprature = float(request.form.get('temprature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # prediction
    crops = ['apple','banana','rice','pomegranate','pigeonpeas','papaya','orange','muskmelon','mungbean','mothbeans','mango','maize',
         'lentil','kidneybeans','jute','grapes','cotton','coffee','coconut','chickpea','blackgram','watermelon']
    result = model.predict(np.array([N, P, K, temprature, humidity, ph, rainfall]).reshape(1,7))
    index = result[0] - 1
    # image = Image.open(f"crops/{crops[index]}.jpg")
    # image.show()
    result = str(crops[index])
    return render_template('crop4.html', result = result)


if __name__ == '__main__':
    app.run(debug = True,port=5001)
