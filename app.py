
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import sqlite3

app = Flask(__name__)
app.secret_key = "medilab_secret_key"

# Developer: Jaspreet Kaur

# Load ML Model
with open("medical_lab_model.pkl", "rb") as file:
    model = pickle.load(file)


# Create contact database
def create_database():
    conn = sqlite3.connect("messages.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# Home and Prediction
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":
        try:
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

        except Exception:
            flash("Prediction failed. Please check your details.", "error")

    return render_template(
        "index.html",
        prediction=prediction
    )


# Contact Form
@app.route("/contact", methods=["POST"])
def contact():

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    message = request.form.get("message", "").strip()

    if not name or not email or not message:
        flash("Please fill all the details.", "error")
        return redirect(url_for("home") + "#contact")

    conn = sqlite3.connect("messages.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (name, email, message) VALUES (?, ?, ?)",
        (name, email, message)
    )

    conn.commit()
    conn.close()

    flash("Your message has been submitted successfully!", "success")

    return redirect(url_for("home") + "#contact")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
