from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///research.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

@app.route('/')
def index():
    return render_template('index.html', 
        page_title="Blockchain Research Assistant",
        welcome_message="Welcome to the blockchain technology research assistant"
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)