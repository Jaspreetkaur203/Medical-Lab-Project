
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import sqlite3

app = Flask(__name__)
app.secret_key = "medilab_secret_key"

# Developer: Jaspreet Kaur

# Load trained ML model
with open("medical_lab_model.pkl", "rb") as file:
    model = pickle.load(file)

# Exact feature columns from your trained model
FEATURES = [
    "age",
    "sex_Male",
    "department_General Medicine",
    "department_Neurology",
    "department_Orthopedics",
    "department_Pediatrics"
]

# Verify model columns
if list(model.feature_names_in_) != FEATURES:
    raise ValueError("Model features do not match app.py")


# Contact database
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
    submitted = None

    if request.method == "POST":
        try:
            age = int(request.form.get("age", ""))
            sex = request.form.get("sex", "")
            department = request.form.get("department", "")

            if not 1 <= age <= 120:
                raise ValueError("Enter age between 1 and 120.")

            if sex not in ["Female", "Male"]:
                raise ValueError("Please select a valid gender.")

            departments = [
                "General Medicine",
                "Neurology",
                "Orthopedics",
                "Pediatrics"
            ]

            if department not in departments:
                raise ValueError("Please select a valid department.")

            # Manual encoding matching the trained model
            input_data = pd.DataFrame([{
                "age": age,
                "sex_Male": 1 if sex == "Male" else 0,
                "department_General Medicine": 1 if department == "General Medicine" else 0,
                "department_Neurology": 1 if department == "Neurology" else 0,
                "department_Orthopedics": 1 if department == "Orthopedics" else 0,
                "department_Pediatrics": 1 if department == "Pediatrics" else 0
            }], columns=FEATURES)

            # ML prediction
            prediction = model.predict(input_data)[0]

            submitted = {
                "age": age,
                "sex": sex,
                "department": department
            }

        except ValueError as error:
            flash(str(error), "error")

        except Exception:
            app.logger.exception("Prediction error")
            flash("Prediction failed. Please try again.", "error")

    return render_template(
        "index.html",
        prediction=prediction,
        submitted=submitted
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
