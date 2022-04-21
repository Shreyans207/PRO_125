from flask import Flask , request, jsonify 
from classifier import getPrediction

app = Flask(__name__)
@app.route('/predict_digit' , methods = ['POST'])

def predict_data() : 
    if not request.json :
        return jsonify({
            'status'  : 'Please submit a valid file',
            'error_code' : 400
        })  
    image = request.files.get('alphabet')
    predicted_value = getPrediction(image)

    return jsonify(predicted_value)

if __name__ == '__main__'  : 
    app.run(debug = True , port = 8080)
    
