import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Load trained models and scaler
try:
    rf_model = joblib.load("random_forest_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    exit()

# Create GUI window
window = tk.Tk()
window.title("Wildfire Prediction System")
window.geometry("600x700")

# Configure style
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 10))
style.configure('TButton', font=('Helvetica', 10, 'bold'))

# Input fields frame
input_frame = ttk.Frame(window, padding="20")
input_frame.pack(fill='both', expand=True)

# Input fields configuration (All 13 features)
features = [
    ('Temperature (°C)', 'temp'),
    ('Rainfall (mm)', 'rain'),
    ('Relative Humidity (%)', 'rh'),
    ('Wind Speed (km/h)', 'ws'),
    ('FFMC', 'ffmc'),
    ('DMC', 'dmc'),
    ('DC', 'dc'),
    ('ISI', 'isi'),
    ('BUI', 'bui'),
    ('FWI', 'fwi')
]

entries = {}

# Create input fields for all features
for i, (label, name) in enumerate(features):
    ttk.Label(input_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = ttk.Entry(input_frame)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky='ew')
    entries[name] = entry


# Prediction function
def predict_fire():
    try:
        # Get input values and validate them
        input_values = []
        for name in ['temp', 'rain', 'rh', 'ws', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']:
            value = entries[name].get()
            if not value.strip():  # Check for empty inputs
                raise ValueError(f"{name} field is empty.")
            input_values.append(float(value))  # Convert to float

        # Convert to array and scale
        input_array = np.array([input_values])  # Use np.array to create array
        scaled_input = scaler.transform(input_array)  # Scale the inputs

        # Make predictions using trained models
        rf_pred = rf_model.predict(scaled_input)[0]
        xgb_pred = xgb_model.predict(scaled_input)[0]

        # Format results
        result_rf = "Fire 🔥" if rf_pred == 1 else "No Fire ✅"
        result_xgb = "Fire 🔥" if xgb_pred == 1 else "No Fire ✅"

        messagebox.showinfo("Prediction Results", f"Random Forest: {result_rf}\nXGBoost: {result_xgb}")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Please enter valid numerical values in all fields!\nError: {ve}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")


# Prediction button
predict_btn = ttk.Button(input_frame, text="Predict Wildfire Risk", command=predict_fire)
predict_btn.grid(row=len(features), columnspan=2, pady=20, sticky='ew')

# Configure grid weights
input_frame.columnconfigure(1, weight=1)

# Run the application
window.mainloop()
