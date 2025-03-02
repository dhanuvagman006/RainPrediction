from flask import Flask, jsonify, request

app = Flask(__name__)p

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask Server!"})

@app.route('/api/data', methods=['GET'])
def get_data():
    sample_data = {"name": "Flask", "type": "Web Framework"}
    return jsonify(sample_data)

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({"received": data}), 201

if __name__ == '__main__':
    app.run(debug=True)