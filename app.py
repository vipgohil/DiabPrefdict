from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define expected feature names (14 features)
feature_names = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'gender_encoded_Female', 'gender_encoded_Male',
    'smoking_history_encoded_No Info', 'smoking_history_encoded_current',
    'smoking_history_encoded_ever', 'smoking_history_encoded_former',
    'smoking_history_encoded_never', 'smoking_history_encoded_not current'
]

def preprocess_input(form_data):
    """Convert user input into the required 14-feature format"""
    try:
        # Convert numeric inputs
        age = float(form_data.get("age", 0))
        hypertension = int(form_data.get("hypertension", 0))
        heart_disease = int(form_data.get("heart_disease", 0))
        bmi = float(form_data.get("bmi", 0))
        HbA1c_level = float(form_data.get("HbA1c_level", 0))
        blood_glucose_level = float(form_data.get("blood_glucose_level", 0))

        # Encode gender
        gender = form_data.get("gender", "Male")  # Default: Male
        gender_encoded = [1, 0] if gender == "Female" else [0, 1]  # [Female, Male]

        # Encode smoking history
        smoking_mapping = {
            "No Info": [1, 0, 0, 0, 0, 0],
            "Current": [0, 1, 0, 0, 0, 0],
            "Ever": [0, 0, 1, 0, 0, 0],
            "Former": [0, 0, 0, 1, 0, 0],
            "Never": [0, 0, 0, 0, 1, 0],
            "Not Current": [0, 0, 0, 0, 0, 1]
        }
        smoking_status = form_data.get("smoking_history", "No Info")
        smoking_encoded = smoking_mapping.get(smoking_status, [1, 0, 0, 0, 0, 0])

        # Combine all features into an array
        input_values = [age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]
        input_values.extend(gender_encoded)  # Add gender encoding
        input_values.extend(smoking_encoded)  # Add smoking encoding

        return np.array(input_values).reshape(1, -1)

    except Exception as e:
        return str(e)  # Return error message

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Process user input
            input_data = preprocess_input(request.form)

            if isinstance(input_data, str):  # If an error message was returned
                return f"Error: {input_data}", 400

            # Make prediction
            prediction = model.predict(input_data)
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

            return render_template("index.html", prediction_text=f"Prediction: {result}")

        except Exception as e:
            return f"Error: {str(e)}", 400  # Display error for debugging

    return render_template("index.html", prediction_text="")

if __name__ == "__main__":
    app.run(debug=True)
