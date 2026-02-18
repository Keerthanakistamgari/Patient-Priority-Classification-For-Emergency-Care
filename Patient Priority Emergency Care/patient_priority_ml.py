"""
Patient Priority Emergency Care - Machine Learning Model
"""

# 1. Introduction
# This script analyzes patient data to predict triage categories in emergency care settings
# using machine learning techniques. It includes data preprocessing, exploratory analysis,
# feature selection, model development, and a GUI for practical application.

# 3. Installing & Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 4. Data Acquisition & Description
def load_data(file_path='patient_priority_extended.csv'):
    """Load the dataset and display basic information"""
    df = pd.read_csv(file_path)
    print("4.1 Data Information:")
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n4.2 Data Statistics:")
    print(df.describe().T)
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nTarget Variable Distribution:")
    print(df['triage'].value_counts())
    
    return df

# 5. Data Pre-processing
def preprocess_data(df):
    """Preprocess the data for analysis and modeling"""
    print("\n5. Data Pre-processing")
    
    # 5.1 Pre-Profiling Report
    print("\n5.1 Pre-Profiling Report:")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # 5.2 Identification & Handling of Missing Data
    print("\n5.2 Handling Missing Data:")
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Numerical columns - fill with median
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("Missing values after handling:\n", df.isnull().sum().sum())
    
    return df

# 6. Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n6. Exploratory Data Analysis")
    
    # Create a directory for plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 6.1 Univariate Analysis
    print("\n6.1 Univariate Analysis")
    
    # Categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/univariate_{col}.png')
        plt.close()
    
    # Numerical variables
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'plots/univariate_{col}.png')
        plt.close()
    
    # 6.2 Bivariate Analysis
    print("\n6.2 Bivariate Analysis")
    
    # Relationship between numerical features and target
    target = 'triage'
    for col in num_cols:
        if col != target:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f'{col} vs {target}')
            plt.tight_layout()
            plt.savefig(f'plots/bivariate_{col}_vs_{target}.png')
            plt.close()
    
    # Relationship between categorical features and target
    for col in cat_cols:
        if col != target:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue=target, data=df)
            plt.title(f'{col} vs {target}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/bivariate_{col}_vs_{target}.png')
            plt.close()
    
    # 6.3 Multivariate Analysis
    print("\n6.3 Multivariate Analysis")
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Pair plot for selected numerical features
    selected_features = num_cols[:5].tolist()  # Select first 5 numerical features
    if target not in selected_features:
        selected_features.append(target)
    
    plt.figure(figsize=(15, 12))
    sns.pairplot(df[selected_features], hue=target)
    plt.savefig('plots/pairplot.png')
    plt.close()
    
    return df

# 7. Post Data Processing & Feature Selection
def feature_engineering(df):
    """Perform feature selection and encoding"""
    print("\n7. Post Data Processing & Feature Selection")
    
    # 7.1 Feature Selection
    print("\n7.1 Feature Selection")
    
    # Separate features and target
    X = df.drop('triage', axis=1)
    y = df['triage']
    
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # 7.2 Encoding the Categorical Data
    print("\n7.2 Encoding Categorical Data")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor, le

# 8. Model Development & Evaluation
def build_models(X_train, X_test, y_train, y_test, preprocessor):
    """Build and evaluate machine learning models"""
    print("\n8. Model Development & Evaluation")
    
    # 8.1 Models Implementation
    print("\n8.1 Models Implementation")
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
    
    # 8.2 Visual Representation of Results
    print("\n8.2 Visual Representation of Results")
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    accuracies = [results[name]['accuracy'] for name in models.keys()]
    sns.barplot(x=list(models.keys()), y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Plot confusion matrices
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
    
    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['pipeline']
    
    print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    
    return best_model, results

# 9. Conclusion
def conclusion(results):
    """Summarize the findings and results"""
    print("\n9. Conclusion")
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"The best performing model is {best_model_name} with an accuracy of {best_accuracy:.4f}")
    print("This model can be used to predict patient triage categories in emergency care settings.")
    print("The model has been saved as 'best_model.pkl' and can be loaded for future predictions.")
    
    return best_model_name, best_accuracy

# 10. GUI Development - Web App Style
class PatientTriageApp:
    def __init__(self, root, model, label_encoder):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder
        self.patient_list = []  # List to store patients
        
        # Configure the window
        self.root.title(" Patient Priority Emergency Care")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f2f5")  # Facebook-like background color
        
        # Set style for a modern look
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#f0f2f5')
        style.configure('TLabel', background='#f0f2f5', font=('Segoe UI', 10))
        style.configure('TLabelframe', background='#f0f2f5')
        style.configure('TLabelframe.Label', font=('Segoe UI', 12, 'bold'), background='#f0f2f5')
        style.configure('Header.TLabel', font=('Segoe UI', 18, 'bold'), background='#f0f2f5')
        style.configure('TButton', font=('Segoe UI', 10), background='#1877f2', foreground='white')
        style.map('TButton', background=[('active', '#166fe5')])
        style.configure('TCombobox', font=('Segoe UI', 10))
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="20", style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Simple header with colored background instead of image
        header_frame = tk.Frame(main_frame, bg="#1877f2", height=100)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 20))
        
        # Make sure the frame maintains its height
        header_frame.grid_propagate(False)
        
        # Add text overlay on the colored background
        header_text = tk.Label(header_frame, text="Patient Priority Emergency Care", 
                             font=("Segoe UI", 24, "bold"), fg="white", bg="#1877f2")
        header_text.place(relx=0.5, rely=0.5, anchor="center")
        
        # Create tabs for different sections
        tab_control = ttk.Notebook(main_frame)
        
        # Tab 1: Patient Registration
        registration_tab = ttk.Frame(tab_control, style='TFrame')
        tab_control.add(registration_tab, text="Patient Registration")
        
        # Tab 2: Patient Queue
        queue_tab = ttk.Frame(tab_control, style='TFrame')
        tab_control.add(queue_tab, text="Patient Queue")
        
        tab_control.grid(row=1, column=0, columnspan=3, sticky="nsew")
        
        # ===== Registration Tab =====
        # Create a frame for the form with a white background and shadow effect
        form_frame = tk.Frame(registration_tab, bg="white", bd=1, relief=tk.SOLID)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title for the form
        tk.Label(form_frame, text="Register New Patient", 
                font=("Segoe UI", 16, "bold"), bg="white").pack(pady=(20, 10))
        
        # Create scrollable frame for form fields
        canvas = tk.Canvas(form_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Define input fields with dropdown options
        self.fields = {
            "name": {"label": "Patient Name", "type": "str", "row": 0, "options": None},
            "age": {"label": "Age", "type": "int", "row": 1, "options": list(range(1, 121))},
            "gender": {"label": "Gender", "type": "int", "row": 2, "options": [(0, "Female"), (1, "Male")]},
            #"chest_pain_type": {"label": "Chest Pain Type", "type": "int", "row": 3, "options": [(1, "Typical Angina"), (2, "Atypical Angina"), (3, "Non-anginal Pain"), (4, "Asymptomatic")]},
            "blood_pressure": {"label": "Blood Pressure", "type": "int", "row": 4, "options": list(range(80, 201, 10))},
            "cholesterol": {"label": "Cholesterol", "type": "int", "row": 5, "options": list(range(100, 601, 20))},
            "max_heart_rate": {"label": "Max Heart Rate", "type": "int", "row": 6, "options": list(range(60, 221, 10))},
            #"exercise_angina": {"label": "Exercise Angina", "type": "int", "row": 7, "options": [(0, "No"), (1, "Yes")]},
            "plasma_glucose": {"label": "Plasma Glucose", "type": "int", "row": 8, "options": list(range(50, 301, 10))},
            #"skin_thickness": {"label": "Skin Thickness", "type": "int", "row": 9, "options": list(range(0, 101, 5))},
            #"insulin": {"label": "Insulin", "type": "int", "row": 10, "options": list(range(0, 851, 50))},
            "bmi": {"label": "BMI", "type": "float", "row": 11, "options": [round(i * 0.1, 1) for i in range(150, 501)]},
            #"diabetes_pedigree": {"label": "Diabetes Pedigree", "type": "float", "row": 12, "options": [round(i * 0.1, 1) for i in range(1, 26)]},
            "hypertension": {"label": "Hypertension", "type": "int", "row": 13, "options": [(0, "No"), (1, "Yes")]},
            "heart_disease": {"label": "Heart Disease", "type": "int", "row": 14, "options": [(0, "No"), (1, "Yes")]},
            #"Residence_type": {"label": "Residence Type", "type": "str", "row": 15, "options": ["Urban", "Rural"]},
            #"smoking_status": {"label": "Smoking Status", "type": "str", "row": 16, "options": ["Never", "Former", "Current"]}
        }
        
        # Create input fields with modern styling
        self.entries = {}
        for field, props in self.fields.items():
            # Create a frame for each field
            field_frame = tk.Frame(scrollable_frame, bg="white", pady=5)
            field_frame.pack(fill="x", padx=20)
            
            # Add label
            tk.Label(field_frame, text=props["label"], bg="white", 
                    font=("Segoe UI", 10), anchor="w").pack(fill="x")
            
            # Add dropdown or entry based on options
            if props["options"] is None:
                # Use regular entry for fields without options (like name)
                entry = tk.Entry(field_frame, font=("Segoe UI", 10), bd=1, relief=tk.SOLID)
                entry.pack(fill="x", pady=(2, 5))
                self.entries[field] = entry
            else:
                # Use combobox for fields with options
                if isinstance(props["options"][0], tuple):
                    # For options with value-label pairs
                    values = [str(label) for value, label in props["options"]]
                    combo = ttk.Combobox(field_frame, values=values, font=("Segoe UI", 10), state="readonly")
                    combo.pack(fill="x", pady=(2, 5))
                    # Store the mapping for later retrieval
                    combo.value_map = {str(label): value for value, label in props["options"]}
                else:
                    # For simple options
                    values = [str(opt) for opt in props["options"]]
                    combo = ttk.Combobox(field_frame, values=values, font=("Segoe UI", 10), state="readonly")
                    combo.pack(fill="x", pady=(2, 5))
                
                self.entries[field] = combo
        
        # Create buttons frame with modern styling
        button_frame = tk.Frame(scrollable_frame, bg="white", pady=15)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        # Add buttons with modern styling
        register_btn = tk.Button(button_frame, text="Register & Predict", bg="#1877f2", fg="white",
                               font=("Segoe UI", 10, "bold"), bd=0, padx=15, pady=8,
                               command=self.register_patient)
        register_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_frame, text="Clear Fields", bg="#e4e6eb", fg="black",
                            font=("Segoe UI", 10), bd=0, padx=15, pady=8,
                            command=self.clear_fields)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        sample_btn = tk.Button(button_frame, text="Load Sample", bg="#e4e6eb", fg="black",
                             font=("Segoe UI", 10), bd=0, padx=15, pady=8,
                             command=self.load_sample)
        sample_btn.pack(side=tk.LEFT, padx=5)
        
        # ===== Queue Tab =====
        # Create a frame for the patient queue with a white background
        queue_frame = tk.Frame(queue_tab, bg="white", bd=1, relief=tk.SOLID)
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title for the queue
        tk.Label(queue_frame, text="Patient Priority Queue", 
                font=("Segoe UI", 16, "bold"), bg="white").pack(pady=(20, 10))
        
        # Create a frame for the treeview
        tree_frame = tk.Frame(queue_frame, bg="white")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create scrollbar for the treeview
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create the treeview
        self.patient_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set)
        self.patient_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar
        tree_scroll.config(command=self.patient_tree.yview)
        
        # Define columns
        self.patient_tree['columns'] = ('name', 'age', 'triage', 'urgency', 'confidence')
        
        # Format columns
        self.patient_tree.column('#0', width=0, stretch=tk.NO)
        self.patient_tree.column('name', anchor=tk.W, width=150)
        self.patient_tree.column('age', anchor=tk.CENTER, width=50)
        self.patient_tree.column('triage', anchor=tk.CENTER, width=100)
        self.patient_tree.column('urgency', anchor=tk.CENTER, width=120)
        self.patient_tree.column('confidence', anchor=tk.CENTER, width=100)
        
        # Create headings
        self.patient_tree.heading('#0', text='', anchor=tk.CENTER)
        self.patient_tree.heading('name', text='Patient Name', anchor=tk.CENTER)
        self.patient_tree.heading('age', text='Age', anchor=tk.CENTER)
        self.patient_tree.heading('triage', text='Triage Color', anchor=tk.CENTER)
        self.patient_tree.heading('urgency', text='Urgency Level', anchor=tk.CENTER)
        self.patient_tree.heading('confidence', text='Confidence', anchor=tk.CENTER)
        
        # Add buttons for queue management
        queue_button_frame = tk.Frame(queue_frame, bg="white", pady=15)
        queue_button_frame.pack(fill="x", padx=20, pady=10)
        
        refresh_btn = tk.Button(queue_button_frame, text="Refresh Queue", bg="#1877f2", fg="white",
                              font=("Segoe UI", 10, "bold"), bd=0, padx=15, pady=8,
                              command=self.refresh_queue)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        remove_btn = tk.Button(queue_button_frame, text="Remove Selected", bg="#e41e3f", fg="white",
                             font=("Segoe UI", 10, "bold"), bd=0, padx=15, pady=8,
                             command=self.remove_patient)
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")
    
    def register_patient(self):
        try:
            # Get values from entries
            patient_data = {}
            for field, props in self.fields.items():
                if field == "name":  # Skip name for prediction but keep for display
                    patient_name = self.entries[field].get()
                    if not patient_name:
                        messagebox.showerror("Error", "Please enter patient name")
                        return
                    continue
                    
                entry = self.entries[field]
                value = entry.get()
                
                if not value:
                    messagebox.showerror("Error", f"Please enter a value for {props['label']}")
                    return
                
                # Handle dropdown values
                if hasattr(entry, 'value_map') and value in entry.value_map:
                    # Get the actual value from the mapping
                    value = entry.value_map[value]
                
                if props["type"] == "int":
                    patient_data[field] = int(value)
                elif props["type"] == "float":
                    patient_data[field] = float(value)
                else:
                    patient_data[field] = value
            
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Make prediction
            prediction = self.model.predict(patient_df)
            prediction_proba = self.model.predict_proba(patient_df)
            
            # Get the predicted class
            predicted_class = self.label_encoder.inverse_transform(prediction)[0]
            
            # Get the probability of the predicted class
            predicted_prob = max(prediction_proba[0]) * 100
            
            # Determine urgency level and color
            if predicted_class == 0 or predicted_class == "red":
                triage_color = "red"
                urgency = "IMMEDIATE"
                priority = 1
            elif predicted_class == 1 or predicted_class == "orange":
                triage_color = "orange"
                urgency = "VERY URGENT"
                priority = 2
            elif predicted_class == 2 or predicted_class == "yellow":
                triage_color = "yellow"
                urgency = "URGENT"
                priority = 3
            else:
                triage_color = "green"
                urgency = "STANDARD"
                priority = 4
            
            # Add patient to the list
            patient_info = {
                'name': patient_name,
                'age': patient_data['age'],
                'triage': triage_color,
                'urgency': urgency,
                'confidence': f"{predicted_prob:.2f}%",
                'priority': priority,
                'data': patient_data
            }
            
            self.patient_list.append(patient_info)
            
            # Sort the patient list by priority
            self.patient_list.sort(key=lambda x: x['priority'])
            
            # Update the treeview
            self.refresh_queue()
            
            # Create a result window to display the prediction
            result_window = tk.Toplevel(self.root)
            result_window.title("Triage Prediction Result")
            result_window.geometry("400x300")
            result_window.configure(bg="white")
            
            # Add a frame with the triage color
            color_frame = tk.Frame(result_window, bg=triage_color, height=50)
            color_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
            
            # Add the triage information
            tk.Label(result_window, text=f"Triage Category: {triage_color.upper()}", 
                   font=("Segoe UI", 16, "bold"), bg="white").pack(pady=(20, 5))
            
            tk.Label(result_window, text=f"Urgency Level: {urgency}", 
                   font=("Segoe UI", 14), bg="white").pack(pady=5)
            
            tk.Label(result_window, text=f"Confidence: {predicted_prob:.2f}%", 
                   font=("Segoe UI", 12), bg="white").pack(pady=5)
            
            # Add a description based on the triage category
            descriptions = {
                "red": "Requires immediate medical attention.",
                "orange": "Very urgent. Should be seen within 10 minutes.",
                "yellow": "Urgent. Should be seen within 1 hour.",
                "green": "Standard. Can wait up to 2-4 hours."
            }
            
            tk.Label(result_window, text=descriptions[triage_color], 
                   font=("Segoe UI", 12), bg="white", wraplength=350).pack(pady=10)
            
            # Add a close button
            tk.Button(result_window, text="Close", bg="#1877f2", fg="white",
                    font=("Segoe UI", 10, "bold"), bd=0, padx=15, pady=8,
                    command=result_window.destroy).pack(pady=20)
            
            # Clear the form
            self.clear_fields()
            
            # Update status
            self.status_var.set(f"Patient {patient_name} registered with {urgency} priority")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def refresh_queue(self):
        # Clear the treeview
        for item in self.patient_tree.get_children():
            self.patient_tree.delete(item)
        
        # Sort the patient list by priority
        self.patient_list.sort(key=lambda x: x['priority'])
        
        # Add patients to the treeview
        for i, patient in enumerate(self.patient_list):
            # Set tag for row color based on triage
            tag = patient['triage']
            
            # Insert patient data
            self.patient_tree.insert(parent='', index='end', iid=i, text='',
                                   values=(patient['name'], patient['age'], 
                                          patient['triage'].upper(), 
                                          patient['urgency'], 
                                          patient['confidence']),
                                   tags=(tag,))
        
        # Configure tag colors
        self.patient_tree.tag_configure('red', background='#ffcccc')
        self.patient_tree.tag_configure('orange', background='#ffeacc')
        self.patient_tree.tag_configure('yellow', background='#ffffcc')
        self.patient_tree.tag_configure('green', background='#ccffcc')
    
    def remove_patient(self):
        # Get selected item
        selected_item = self.patient_tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a patient to remove")
            return
        
        # Get the index of the selected item
        index = int(selected_item[0])
        
        # Remove the patient from the list
        patient_name = self.patient_list[index]['name']
        self.patient_list.pop(index)
        
        # Update the treeview
        self.refresh_queue()
        
        # Update status
        self.status_var.set(f"Patient {patient_name} removed from queue")
    
    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.status_var.set("Form cleared")
    
    def load_sample(self):
        # Load a sample patient from the dataset
        try:
            df = pd.read_csv('patient_priority_extended.csv')
            sample = df.sample(1).iloc[0]
            
            # Generate a random name
            import random
            names = ["John Smith", "Maria Garcia", "David Lee", "Sarah Johnson", 
                    "Michael Brown", "Emma Wilson", "James Taylor", "Olivia Martinez"]
            random_name = random.choice(names)
            
            # Fill the entries with sample data
            self.entries["name"].delete(0, tk.END)
            self.entries["name"].insert(0, random_name)
            
            for field in self.fields:
                if field != "name" and field in sample:
                    self.entries[field].delete(0, tk.END)
                    self.entries[field].insert(0, str(sample[field]))
            
            self.status_var.set("Sample patient data loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load sample data: {str(e)}")

# Main function to run the entire pipeline
def main():
    # Check if we should skip the ML pipeline and just launch the GUI
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--gui-only":
        print("\n10. Launching GUI Only")
        # Load the saved model and label encoder
        try:
            best_model = joblib.load('best_model.pkl')
            # Create a simple label encoder for the triage categories
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['green', 'yellow', 'orange', 'red'])
            
            # Launch GUI
            root = tk.Tk()
            app = PatientTriageApp(root, best_model, label_encoder)
            root.mainloop()
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to full pipeline...")
    
    # Regular pipeline with ML training
    # Load data
    df = load_data()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform EDA
    df = perform_eda(df)
    
    # Feature engineering
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = feature_engineering(df)
    
    # Build and evaluate models
    best_model, results = build_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Conclusion
    best_model_name, best_accuracy = conclusion(results)
    
    # Launch GUI
    print("\n10. Launching GUI")
    root = tk.Tk()
    app = PatientTriageApp(root, best_model, label_encoder)
    root.mainloop()

if __name__ == "__main__":
    main()