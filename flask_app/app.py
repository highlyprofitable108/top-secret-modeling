from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = '4815162342'  # Change this to a random secret key


@app.route('/')
def home():
    return render_template('flask_app/index.html')


@app.route('/select_stats', methods=['GET', 'POST'])
def select_stats():
    if request.method == 'POST':
        # Here, you will handle the form submission and interact with nfl_select_stats.py
        pass
    return render_template('select_stats.html')



if __name__ == "__main__":
    app.run(debug=True)
