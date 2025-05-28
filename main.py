from flask import Flask 
from descent import descent
from scikit import scikit
from ridge import ridge
from lasso import lasso

app =Flask(__name__)
app.register_blueprint(scikit,url_prefix="")
app.register_blueprint(descent,url_prefix="")
app.register_blueprint(ridge ,url_prefix="")
app.register_blueprint(lasso ,url_prefix="")

@app.route("/")
def home():
    return "WELCOME TO PREDICTION"

if __name__ == "__main__":
    app.run(debug=True)