
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import sys
import os

# Mock the model and encoder
class MockModel:
    def predict(self, X):
        return [0]
    def predict_proba(self, X):
        return [[1, 0, 0, 0]]

class MockEncoder:
    def inverse_transform(self, arr):
        return ["green"]

# Import the class from the file
sys.path.append(os.getcwd())
from launch_gui import PatientTriageApp

def test_validation():
    root = tk.Tk()
    app = PatientTriageApp(root, MockModel(), MockEncoder())
    
    # Mock entries
    app.entries['name'].insert(0, "Test Patient")
    app.entries['age'].insert(0, "150") # Above limit (0-120)
    app.entries['blood pressure'].insert(0, "120")
    app.entries['cholesterol'].insert(0, "180")
    app.entries['max heart rate'].insert(0, "80")
    app.entries['plasma glucose'].insert(0, "100")
    app.entries['bmi'].insert(0, "25")
    
    # Mock messagebox.showerror to capture the error
    original_showerror = messagebox.showerror
    error_called = []
    
    def mock_showerror(title, message):
        print(f"Error displayed: {title} - {message}")
        error_called.append((title, message))
        
    messagebox.showerror = mock_showerror
    
    # Trigger registration
    print("Testing age > 120...")
    app.register_patient()
    
    if error_called:
        print("Validation caught the error.")
    else:
        print("Validation FAILED to catch the error.")

    # Reset and test valid
    error_called.clear()
    app.entries['age'].delete(0, tk.END)
    app.entries['age'].insert(0, "50")
    
    print("Testing valid age...")
    app.register_patient()
    
    if not error_called:
        print("Validation passed for valid input.")
    else:
        print("Validation erroneously blocked valid input.")

    root.destroy()

if __name__ == "__main__":
    test_validation()
