from flask import Flask, jsonify, request,render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    if request.method=='POST':
        data=request.form.get('username')
        return jsonify({"message": "Welcome to the Flask Server!"})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)