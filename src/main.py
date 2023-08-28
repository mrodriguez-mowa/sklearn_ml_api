from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Mowa's API to classify text :D"

@app.route("/api/process-nlp", methods=['POST'])
def process_messages():
    current_path = os.path.dirname(__file__)

    vectorizer = joblib.load(os.path.join(current_path, "../files/vectorizers/vectorizer.joblib"))
    model = joblib.load(os.path.join(current_path, "../files/models/model.joblib"))

    body = request.get_json()
    response = []

    for element in body['data']:
        message = element['textMessage']
        X = vectorizer.transform([message.lower()])
        y_pred = model.predict(X)
        element['qualification'] = y_pred[0]
        response.append(element)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)