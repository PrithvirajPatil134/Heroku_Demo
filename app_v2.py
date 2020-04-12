import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # final_features = [1]
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # final_features = final_features.append([1])
    # final_features = np.linalg.svd(int_features, full_matrices=False)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Car Price should be $ {} in thousands'.format(output))


if __name__ == "__main__":
    app.run(debug=True)