# Import the necessary modules
from flask import Flask, render_template, request
import pickle

# Create a new instance of the Flask class and assign it to the variable 'app'
app = Flask(__name__)

# Load the trained model from the .pkl file
model = pickle.load(open("student_employability.pkl", "rb"))

# Create a route for the homepage
@app.route("/", methods=["GET", "POST"])
def home():
    # Set default prediction value and empty form_data dictionary
    prediction = "unknown"
    form_data = {}

    # If a form is submitted, update form_data with the new values
    if request.method == "POST":
        form_data["general_appearance"] = int(request.form["gen_app"])
        form_data["manner_of_speaking"] = int(request.form["manner_spk"])
        form_data["physical_condition"] = int(request.form["phys_cond"])
        form_data["mental_alertness"] = int(request.form["ment_alert"])
        form_data["self_confidence"] = int(request.form["self_conf"])
        form_data["ability_to_present_ideas"] = int(request.form["ability_pres_ideas"])
        form_data["communication_skill"] = int(request.form["comm_skill"])
        form_data["student_performance_rating"] = int(request.form["stud_perf_rate"])

        # Make a prediction for the new student using the trained classifier
        prediction = model.predict([list(form_data.values())])[0]

    # Render the index.html template and pass the prediction and form data as arguments
    return render_template("index.html", prediction=prediction, form_data=form_data)

# Run the app only when this script is executed directly (not when imported)
if __name__ == "__main__":
    # Run the app in debug mode, which allows us to see error messages in the browser
    app.run(debug=True)
