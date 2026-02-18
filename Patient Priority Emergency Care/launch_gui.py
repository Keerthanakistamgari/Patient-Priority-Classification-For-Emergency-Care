import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
# ---------------- Fallbacks ----------------
class RuleBasedModel:
    """
    Simple rule-based fallback that assigns triage based on key vitals.
    Returns integer labels: 0=green,1=yellow,2=orange,3=red
    """
    def predict(self, X):
        import numpy as _np
        preds = []
        for _, row in X.iterrows():
            score = 0
            age = float(row.get('age', 0))
            bp = float(row.get('blood pressure', 0))
            chol = float(row.get('cholesterol', 0))
            hr = float(row.get('max heart rate', 0))
            glucose = float(row.get('plasma glucose', 0))
            bmi = float(row.get('bmi', 0))
            htn = int(row.get('hypertension', 0))
            hd = int(row.get('heart_disease', 0))
            # strong red conditions
            if bp < 90 or bp > 180: score += 3
            if hr > 140 or hr < 40: score += 3
            if glucose > 300: score += 3
            # urgent indicators
            if age >= 75: score += 2
            if chol > 300: score += 1
            if bmi > 40: score += 1
            if htn == 1: score += 1
            if hd == 1: score += 1
            # map score to class
            if score >= 5:
                preds.append(3)  # red
            elif score >= 3:
                preds.append(2)  # orange
            elif score >= 1:
                preds.append(1)  # yellow
            else:
                preds.append(0)  # green
        return _np.array(preds, dtype=int)
    def predict_proba(self, X):
        import numpy as _np
        p = []
        labels = [0,1,2,3]
        preds = self.predict(X)
        for pr in preds:
            probs = _np.full(len(labels), 0.0)
            probs[pr] = 0.9
            probs = probs + 0.1/len(labels)
            p.append(probs)
        return _np.array(p)
class DummyLabelEncoder:
    """Fallback encoder mapping indices to triage names."""
    def __init__(self, classes=None):
        self.classes_ = np.array(classes or ['green', 'yellow', 'orange', 'red'])
    def inverse_transform(self, arr):
        return [self.classes_[int(x)] for x in arr]
# ---------------- Application ----------------
class PatientTriageApp:
    """
    Minimal GUI showing only: age, blood pressure, cholesterol,
    max heart rate, plasma glucose, bmi, hypertension, heart_disease.
    """
    def __init__(self, root, model, label_encoder):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder
        self.patient_list = []
        self.root.title("Patient Priority Emergency Care")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        # Use a theme that respects custom row background colors
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        header = tk.Label(main, text="Patient Priority Emergency Care",
                          fg="white", bg="#1877f2",
                          font=("Segoe UI", 20, "bold"), height=2)
        header.pack(fill=tk.X)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        self.reg_frame = ttk.Frame(self.notebook, padding=10)
        self.queue_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.reg_frame, text="Patient Registration")
        self.notebook.add(self.queue_frame, text="Patient Queue")
        self._build_form()
        self._build_queue()
        # status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(fill=tk.X, side=tk.BOTTOM)
    def _build_form(self):
        # Only requested features
        self.fields = [
            "name",
            "age",
            "blood pressure",
            "cholesterol",
            "max heart rate",
            "plasma glucose",
            "bmi",
            "hypertension",
            "heart_disease"
        ]
        self.field_options = {
            "hypertension": ["0", "1"],
            "heart_disease": ["0", "1"]
        }      
        # Range limits for validation
        self.range_limits = {
            "age": (0, 120),
            "blood pressure": (50, 250),
            "cholesterol": (100, 600),
            "max heart rate": (40, 220),
            "plasma glucose": (50, 500),
            "bmi": (10, 60)
        }
        # Display ranges for input parameters
        self.field_ranges = {k: f"({v[0]}-{v[1]})" for k, v in self.range_limits.items()}
        # Larger fonts for labels and inputs in the form
        try:
            form_style = ttk.Style()
            form_style.configure("Form.TLabel", font=("Segoe UI", 12))
            form_style.configure("Form.TEntry", font=("Segoe UI", 12))
            form_style.configure("Form.TCombobox", font=("Segoe UI", 12))
        except Exception:
            pass
        top = ttk.Frame(self.reg_frame)
        top.pack(fill=tk.X, pady=4)
        # Button styles (do not interfere with background image)
        try:
            form_style = ttk.Style()
            form_style.configure("Primary.TButton", background="#4CAF50", foreground="white")
            form_style.map("Primary.TButton", background=[("active", "#43A047")])
            form_style.configure("Danger.TButton", background="#f44336", foreground="white")
            form_style.map("Danger.TButton", background=[("active", "#e53935")])
            form_style.configure("Info.TButton", background="#2196F3", foreground="white")
            form_style.map("Info.TButton", background=[("active", "#1E88E5")])
            form_style.configure("Accent.TButton", background="#FF9800", foreground="white")
            form_style.map("Accent.TButton", background=[("active", "#FB8C00")])
        except Exception:
            pass
        ttk.Button(top, text="Register Patient", style="Primary.TButton",
                   command=self.register_patient).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Clear Form", style="Danger.TButton",
                   command=self.clear_fields).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load Sample", style="Info.TButton",
                   command=self.load_sample).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load Color Samples", style="Accent.TButton",
                   command=self.load_color_samples).pack(side=tk.LEFT, padx=6)
        # Container with background image
        form_container = ttk.LabelFrame(self.reg_frame, text="Patient Information")
        form_container.pack(fill=tk.BOTH, expand=True, pady=8)
        # Colorful gradient background layer (resizes with container)
        self.form_bg_canvas = tk.Canvas(form_container, highlightthickness=0, bd=0)
        self.form_bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Push the background canvas behind other widgets
        try:
            self.form_bg_canvas.lower()
        except Exception:
            pass
        self._init_form_bg_gradient(form_container)
        # Foreground form grid on top of background
        form = ttk.Frame(form_container)
        form.pack(fill=tk.BOTH, expand=True)
        try:
            # Make input columns expand with the container
            form.columnconfigure(1, weight=1)
            form.columnconfigure(3, weight=1)
        except Exception:
            pass
        self.entries = {}
        row = 0
        col = 0
        for field in self.fields:
            label_text = field.replace("_", " ").capitalize()
            if hasattr(self, 'field_ranges') and field in self.field_ranges:
                label_text += f" {self.field_ranges[field]}"
            label_text += ":"           
            lbl = ttk.Label(form, text=label_text, style="Form.TLabel")
            lbl.grid(row=row, column=col, sticky=tk.W, padx=6, pady=6)
            if field in self.field_options:
                ent = ttk.Combobox(form, values=self.field_options[field], state="readonly", width=30, style="Form.TCombobox")
                ent.current(0)
            else:
                # Widen text entry fields; make Name widest
                w = 36 if field == "name" else 32
                ent = ttk.Entry(form, width=w, style="Form.TEntry")
            # Stretch entries horizontally to use available space
            ent.grid(row=row, column=col+1, sticky=tk.EW, padx=6, pady=6)
            self.entries[field] = ent
            col += 2
            if col >= 4:
                col = 0
                row += 1
    def _init_form_bg_gradient(self, container):
        # Draw a colorful gradient background that resizes with the container
        self._form_bg_tk = None
        try:
            from PIL import Image, ImageTk, ImageColor  # type: ignore
            use_pil = True
        except Exception:
            use_pil = False
        # Vibrant color stops from warm to cool
        color_stops = ["#ff6f61", "#ffcc00", "#4caf50", "#2196f3", "#9c27b0"]
        def draw_gradient(event=None):
            w = max(100, container.winfo_width())
            h = max(100, container.winfo_height())
            self.form_bg_canvas.delete("all")
            if use_pil:
                # High-quality horizontal gradient using PIL
                try:
                    img = Image.new("RGB", (w, h))
                    n = len(color_stops)
                    rgbs = [ImageColor.getrgb(c) for c in color_stops]
                    # Build vertical bands efficiently
                    for x in range(w):
                        f = x / max(1, (w - 1))
                        seg = f * (n - 1)
                        i = int(seg)
                        t = seg - i
                        if i >= n - 1:
                            c = rgbs[-1]
                        else:
                            c = tuple(int((1 - t) * rgbs[i][k] + t * rgbs[i + 1][k]) for k in range(3))
                        img.paste(Image.new("RGB", (1, h), c), (x, 0))
                    self._form_bg_tk = ImageTk.PhotoImage(img)
                    self.form_bg_canvas.create_image(0, 0, image=self._form_bg_tk, anchor="nw")
                except Exception:
                    # Fallback to canvas stripes if PIL drawing fails mid-run
                    _draw_canvas_stripes(w, h)
            else:
                _draw_canvas_stripes(w, h)
        def _draw_canvas_stripes(w, h):
            # Gradient approximation with canvas stripes for environments without PIL
            def hex_to_rgb(s):
                s = s.lstrip('#')
                return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
            n = len(color_stops)
            rgbs = [hex_to_rgb(c) for c in color_stops]
            bands = max(32, min(256, w // 6))
            for b in range(bands):
                f = b / max(1, (bands - 1))
                seg = f * (n - 1)
                i = int(seg)
                t = seg - i
                if i >= n - 1:
                    c = rgbs[-1]
                else:
                    c = tuple(int((1 - t) * rgbs[i][k] + t * rgbs[i + 1][k]) for k in range(3))
                color = "#%02x%02x%02x" % c
                x0 = int(b * w / bands)
                x1 = int((b + 1) * w / bands)
                self.form_bg_canvas.create_rectangle(x0, 0, x1, h, fill=color, outline=color)
        container.bind("<Configure>", draw_gradient)
        container.update_idletasks()
        draw_gradient()
    def _build_queue(self):
        # Paned layout: left (queue table), right (image panel)
        pw = ttk.Panedwindow(self.queue_frame, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, pady=8)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw, width=320)
        pw.add(left, weight=3)
        pw.add(right, weight=1)
        table_frame = ttk.LabelFrame(left, text="Patient Queue (Priority Order)")
        table_frame.pack(fill=tk.BOTH, expand=True)
        # Only show the important display columns
        cols = ("name", "age", "triage", "urgency")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        self.tree.heading("name", text="Name")
        self.tree.heading("age", text="Age")
        self.tree.heading("triage", text="Triage")
        self.tree.heading("urgency", text="Urgency Level")
        self.tree.column("name", width=320)
        self.tree.column("age", width=80, anchor=tk.CENTER)
        self.tree.column("triage", width=120, anchor=tk.CENTER)
        self.tree.column("urgency", width=120, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sb.set)
        # Configure tag colors for triage rows (ensure visible backgrounds)
        self.tree.tag_configure('red', background='#ffcccc')
        self.tree.tag_configure('orange', background='#ffeacc')
        self.tree.tag_configure('yellow', background='#ffffcc')
        self.tree.tag_configure('green', background='#ccffcc')
        # Right-side image panel
        img_frame = ttk.LabelFrame(right, text="Info")
        img_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(img_frame, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._load_side_image()
        btns = ttk.Frame(self.queue_frame)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Refresh Queue", command=self.refresh_queue).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Remove Selected", command=self.remove_patient).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Triage Legend", command=self.show_legend).pack(side=tk.LEFT, padx=6)
    def _load_side_image(self):
        # Try to load JPEG with Pillow; fallback to PNGs with tk.PhotoImage
        self._side_img = None
        try:
            from PIL import Image, ImageTk  # type: ignore
            pil_ok = True
        except Exception:
            pil_ok = False
        try:
            if pil_ok and os.path.exists("emergency_room_bg.jpg"):
                im = Image.open("emergency_room_bg.jpg")
                target_w = 300
                target_h = max(200, int(im.height * target_w / im.width))
                im = im.resize((target_w, target_h), Image.LANCZOS)
                self._side_img = ImageTk.PhotoImage(im)
                self.image_label.configure(image=self._side_img)
                return
        except Exception:
            pass
        # PNG fallback options from plots
        fallback_pngs = [
            "plots/correlation_matrix.png",
            "plots/univariate_triage.png",
            "plots/model_comparison.png",
        ]
        for p in fallback_pngs:
            if os.path.exists(p):
                try:
                    self._side_img = tk.PhotoImage(file=p)
                    self.image_label.configure(image=self._side_img)
                    return
                except Exception:
                    continue
        # If everything fails, show text
        self.image_label.configure(text="Image not available", anchor=tk.CENTER)
    def clear_fields(self):
        for name, widget in self.entries.items():
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.current(0)
                else:
                    widget.delete(0, tk.END)
            except Exception:
                pass
        self.status_var.set("Form cleared")
    def load_sample(self):
        # Try to load one sample row from CSV; otherwise populate sensible defaults
        try:
            df = pd.read_csv("patient_priority_extended.csv")
            sample = df.sample(1).iloc[0]
            for f, w in self.entries.items():
                if f == "name":
                    w.delete(0, tk.END); w.insert(0, "Demo Patient")
                    continue
                if f in sample.index:
                    val = sample[f]
                    if isinstance(w, ttk.Combobox):
                        vals = w.cget("values")
                        sval = str(val)
                        if sval in vals:
                            w.set(sval)
                        else:
                            w.set(vals[0] if vals else "")
                    else:
                        w.delete(0, tk.END); w.insert(0, str(val))
                else:
                    # leave empty or defaults
                    if isinstance(w, ttk.Combobox):
                        w.current(0)
                    else:
                        w.delete(0, tk.END)
            self.status_var.set("Sample loaded")
        except Exception:
            # defaults
            defaults = {"age":"30","blood pressure":"120","cholesterol":"180","max heart rate":"90",
                        "plasma glucose":"90","bmi":"25","hypertension":"0","heart_disease":"0"}
            for f,w in self.entries.items():
                if f == "name":
                    w.delete(0, tk.END); w.insert(0, "Demo Patient"); continue
                if f in defaults:
                    if isinstance(w, ttk.Combobox):
                        try: w.set(defaults[f])
                        except: w.current(0)
                    else:
                        w.delete(0, tk.END); w.insert(0, defaults[f])
                else:
                    if isinstance(w, ttk.Combobox): w.current(0)
                    else: w.delete(0, tk.END)
            self.status_var.set("Sample defaults loaded")
    def load_color_samples(self):
        # Insert four demo patients to showcase row colors (red, orange, yellow, green)
        urgency_map = {"red": 1, "orange": 2, "yellow": 3, "green": 4}
        samples = [
            {"name": "Sarah Johnson", "age": "57", "triage": "red"},
            {"name": "David Lee", "age": "66", "triage": "orange"},
            {"name": "Olivia Martinez", "age": "34", "triage": "yellow"},
            {"name": "Michael Brown", "age": "41", "triage": "green"},
        ]
        for s in samples:
            s["urgency"] = urgency_map.get(s["triage"], 4)
        # Avoid duplicates: only add if not present by name and triage
        existing = {(p.get("name"), p.get("triage")) for p in self.patient_list}
        for s in samples:
            key = (s["name"], s["triage"])
            if key not in existing:
                self.patient_list.append(s)
        self.patient_list.sort(key=lambda x: x["urgency"]) 
        self.refresh_queue()
        self.status_var.set("Loaded demo patients for all triage colors")
    def _validate_numeric(self, value, fname):
        try:
            float(value)
            return True, ""
        except Exception:
            return False, f"{fname} must be numeric"
    def register_patient(self):
        pdata = {}
        for f in self.fields:
            val = self.entries[f].get().strip()
            if f != "name" and val == "":
                messagebox.showerror("Error", f"Enter {f}")
                return
            pdata[f] = val
        # Validate numeric fields
        numeric_fields = ["age", "blood pressure", "cholesterol", "max heart rate", "plasma glucose", "bmi"]
        for nf in numeric_fields:
            ok, msg = self._validate_numeric(pdata.get(nf, ""), nf)
            if not ok:
                messagebox.showerror("Validation error", msg)
                return          
            # Check ranges
            if hasattr(self, 'range_limits') and nf in self.range_limits:
                val = float(pdata.get(nf))
                min_val, max_val = self.range_limits[nf]
                if not (min_val <= val <= max_val):
                    messagebox.showerror("Validation error", 
                        f"{nf.replace('_',' ').capitalize()} must be between {min_val} and {max_val}")
                    return
        # Prepare input dataframe with selected features
        try:
            input_df = pd.DataFrame({
                'age': [float(pdata['age'])],
                'blood pressure': [float(pdata['blood pressure'])],
                'cholesterol': [float(pdata['cholesterol'])],
                'max heart rate': [float(pdata['max heart rate'])],
                'plasma glucose': [float(pdata['plasma glucose'])],
                'bmi': [float(pdata['bmi'])],
                'hypertension': [int(pdata['hypertension'])],
                'heart_disease': [int(pdata['heart_disease'])]
            })
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare input data: {e}")
            return
        # Predict (safe). If model returns a single constant class or very low confidence,
        # use the rule-based fallback.
        triage = None
        pred_label = None
        probs = None
        try:
            pred = self.model.predict(input_df)
            pred_label = int(pred[0])
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(input_df)
            # detect degenerate model: single unique prediction OR all probs nearly equal
            degenerate = False
            try:
                import numpy as _np
                if getattr(self.model, '__class__', None).__name__ == 'RuleBasedModel':
                    degenerate = False
                else:
                    # check predict output uniqueness
                    if _np.unique(pred).size == 1:
                        degenerate = True
                    # check probabilities uniform/low confidence
                    if probs is not None:
                        if float(_np.max(probs[0])) < 0.45:
                            degenerate = True
            except Exception:
                degenerate = False
            if degenerate:
                # fallback to rule-based
                rb = RuleBasedModel()
                pred_rb = rb.predict(input_df)
                pred_label = int(pred_rb[0])
                probs = rb.predict_proba(input_df)
        except Exception:
            # model totally failed -> use rule-based
            rb = RuleBasedModel()
            pred_rb = rb.predict(input_df)
            pred_label = int(pred_rb[0])
            probs = rb.predict_proba(input_df)
        # map label to triage string via encoder or fallback mapping
        try:
            triage = self.label_encoder.inverse_transform([pred_label])[0]
        except Exception:
            mapping = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
            triage = mapping.get(pred_label, 'green')

        urgency_map = {"red":1, "orange":2, "yellow":3, "green":4}
        urgency = urgency_map.get(triage, 4)
        entry = {
            "name": pdata.get("name",""),
            "age": pdata.get("age",""),
            "triage": triage,
            "urgency": urgency
        }
        self.patient_list.append(entry)
        self.patient_list.sort(key=lambda x: x["urgency"])
        self.refresh_queue()
        # show result popup
        popup = tk.Toplevel(self.root)
        popup.title("Triage Result")
        popup.geometry("360x220")
        ttk.Label(popup, text=f"Patient: {entry['name']}").pack(pady=6)
        ttk.Label(popup, text=f"Triage: {triage.upper()}").pack(pady=6)
        ttk.Label(popup, text=f"Urgency level: {urgency}").pack(pady=6)
        # Show class probabilities if available
        try:
            prob_lines = []
            if probs is not None:
                classes = getattr(self.label_encoder, 'classes_', None)
                if classes is not None and len(classes) == probs.shape[1]:
                    prob_map = {str(classes[i]).lower(): float(probs[0][i]) for i in range(len(classes))}
                    display_order = ['red', 'orange', 'yellow', 'green']
                    for cls in display_order:
                        if cls in prob_map:
                            prob_lines.append(f"{cls.capitalize()}: {prob_map[cls]*100:.1f}%")
                    # If some classes not in display_order, append remaining
                    for k in prob_map.keys():
                        if k not in display_order:
                            prob_lines.append(f"{k.capitalize()}: {prob_map[k]*100:.1f}%")
                else:
                    # Fallback: show raw probabilities
                    for i in range(probs.shape[1]):
                        prob_lines.append(f"Class {i}: {probs[0][i]*100:.1f}%")
            if prob_lines:
                ttk.Label(popup, text="Class probabilities:").pack(pady=(8,2))
                for line in prob_lines:
                    ttk.Label(popup, text=line).pack()
        except Exception:
            pass
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)

        self.status_var.set(f"Registered {entry['name']} => {triage.upper()}")
    def refresh_queue(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for p in self.patient_list:
            tag = p.get("triage", "").lower()
            self.tree.insert(
                "", tk.END,
                values=(p.get("name",""), p.get("age",""), str(p.get("triage","")).upper(), p.get("urgency","")),
                tags=(tag,)
            )
    def remove_patient(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a patient to remove")
            return
        idx = self.tree.index(sel[0])
        if 0 <= idx < len(self.patient_list):
            removed = self.patient_list.pop(idx)
            self.refresh_queue()
            self.status_var.set(f"Removed {removed['name']}")
    def show_legend(self):
        msg = (
            "Triage colors and urgency levels:\n\n"
            "Red = life-threatening (urgency 1)\n"
            "Orange = high urgency (urgency 2)\n"
            "Yellow = moderate priority (urgency 3)\n"
            "Green = routine care (urgency 4)\n\n"
            "Predictions select the class with the highest probability.\n"
            "You can see per-class probabilities in the result popup."
        )
        try:
            messagebox.showinfo("Triage Legend", msg)
        except Exception:
            pass
# ---------------- Main ----------------
def main():
    use_demo = ('--no-model' in sys.argv or '--demo' in sys.argv)
    model = None
    label_encoder = None
    model_files = ["best_model.pkl", "model.pkl"]
    le_files = ["label_encoder.pkl"]
    if use_demo:
        model = RuleBasedModel()
        label_encoder = DummyLabelEncoder()
        print("Demo mode: using fallback model and encoder")
    else:
        for p in model_files:
            if os.path.exists(p):
                try:
                    model = joblib.load(p)
                    print(f"Loaded model from {p}")
                    break
                except Exception as e:
                    print(f"Failed to load model {p}: {e}")
        if model is None:
            print("No model file found or load failed. Using demo model.")
            model = DummyModel()
        for p in le_files:
            if os.path.exists(p):
                try:
                    label_encoder = joblib.load(p)
                    print(f"Loaded label encoder from {p}")
                    break
                except Exception as e:
                    print(f"Failed to load label encoder {p}: {e}")
        if label_encoder is None:
            label_encoder = DummyLabelEncoder()
            print("Using demo label encoder")
    root = tk.Tk()
    app = PatientTriageApp(root, model, label_encoder)
    root.mainloop()
if __name__ == "__main__":
    main()
