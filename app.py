
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("medical_lab_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        age = int(request.form["age"])
        sex = request.form["sex"]
        department = request.form["department"]

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "department": department
        }])

        input_data = pd.get_dummies(
            input_data,
            columns=["sex", "department"],
            drop_first=True,
            dtype=int
        )

        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
