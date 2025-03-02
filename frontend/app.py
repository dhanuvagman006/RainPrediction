from flask import Flask, jsonify, request, render_template
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    weather_data = "Rain"
    if request.method == 'POST':
        city = request.form.get('city')
        if city:

    return render_template('index.html', weather=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
