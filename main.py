import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_concrete

app= Flask("Concrete_prediction")

@app.route('/', methods=["POST"])

def predict():
    concret_config = request.get_json()

    with open ("./model_files/model.bin","rb") as read_f:

        model = pickle.load(read_f)
        
        read_f.close()
    predictions = predict_concrete(concret_config,model)

    response ={'Concrete_prediction':list(predictions)}

    return jsonify(response)

# app = Flask("Concrete Prediction")

# @app.route('/',methods=['GET'])

# def ping():
#     return "Prediction concrete!!"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=9696)