from flask import Flask, request, jsonify
import numpy as np
import joblib
import psycopg2

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Docker!"

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({"prediction": int(prediction[0])})

@app.route("/test_db", methods=["GET"])
def test_db():
    try:
        conn = psycopg2.connect(
            host="db",  # Nazwa serwisu z docker-compose
            database="mydatabase",
            user="postgres",
            password="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        conn.close()
        return jsonify({"db_status": "connected", "result": result})
    except Exception as e:
        return jsonify({"db_status": "error", "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)