import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import os
import joblib

from src.models.performance import eval_perform, plot_cm, plot_regression, plot_true_false
from src.models import RegressionModel, BClassifier, get_data


class FootballModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Model Trainer")
        self.root.geometry("800x600")
        
        self.model_type = tk.StringVar(value="regression")
        self.verbose = tk.BooleanVar(value=False)
        
        self.create_widgets()
        self.model = None
        self.metrics = None
        self.model_loaded_from_disk = False

    def create_widgets(self):
        frame_top = ttk.Frame(self.root)
        frame_top.pack(pady=10, fill=tk.X)
        
        ttk.Label(frame_top, text="Select Model Type:").pack(side=tk.LEFT, padx=5)
        model_choices = ("regression", "b_classification")
        self.model_dropdown = ttk.Combobox(frame_top, values=model_choices, textvariable=self.model_type, state="readonly")
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.verbose_check = ttk.Checkbutton(frame_top, text="Verbose Logging", variable=self.verbose)
        self.verbose_check.pack(side=tk.LEFT, padx=20)
        
        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(pady=10)
        
        self.train_btn = ttk.Button(frame_buttons, text="Train Model", command=self.train_model)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.evaluate_btn = ttk.Button(frame_buttons, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot_conf_btn = ttk.Button(frame_buttons, text="Plot Confusion Matrix", command=self.plot_confusion)
        self.plot_conf_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot_bar_btn = ttk.Button(frame_buttons, text="Plot graph", command=self.plot_metrics_bar)
        self.plot_bar_btn.pack(side=tk.LEFT, padx=5)
        
        self.log_text = ScrolledText(self.root, height=20)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.save_btn = ttk.Button(frame_buttons, text="Save Model", command=self.save_model)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.load_btn = ttk.Button(frame_buttons, text="Load Model", command=self.load_model)
        self.load_btn.pack(side=tk.LEFT, padx=5)

    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def train_model(self):
        threading.Thread(target=self._train_model, daemon=True).start()
    
    def _train_model(self):
        self.log("Starting training...")
        model_type = self.model_type.get()
        try:
            if model_type == "regression":
                self.model = RegressionModel()
            elif model_type == "b_classification":
                self.model = BClassifier()
            else:
                raise ValueError("Unknown model type selected.")
                
            x_train, x_test, y_train, y_test = self.model.fit(verbose=self.verbose.get()) \
                if hasattr(self.model, "fit") else (None, None, None, None)
            
            self.log(f"Training complete.\nTrain set size: {x_train.shape[0]}, Test set size: {x_test.shape[0]}")
            if self.verbose.get():
                self.log("Verbose mode enabled: Detailed training logs available.")
        except Exception as e:
            self.log(f"Error during training: {str(e)}")
            messagebox.showerror("Training Error", str(e))
    
    def evaluate_model(self):
        threading.Thread(target=self._evaluate_model, daemon=True).start()
    
    def _evaluate_model(self):
        if self.model is None:
            self.log("Model has not been trained or loaded!")
            return
        
        try:
            if self.model_loaded_from_disk:
                self.log("Using loaded model for evaluation.")
                x_train, x_test, y_train, y_test, _ = get_data(split_type=self.model_type.get())
                pred = self.model.model.predict(x_test)
            else:
                pred, y_test = self.model.get_test_pred()

            self.metrics = eval_perform(pred, y_test)
            self.metrics["y_pred"] = pred 

            self.log("Performance Report:")
            report_lines = [
                f"{key}: {value}" for key, value in self.metrics.items()
                if key != "confusion_matrix"
            ]
            self.log("\n".join(report_lines))

            self.log("Confusion Matrix:")
            self.log(str(self.metrics.get("confusion_matrix", "Not available")))

        except Exception as e:
            self.log(f"Error during evaluation: {str(e)}")
            messagebox.showerror("Evaluation Error", str(e))

    
    def plot_confusion(self):
        if self.metrics is None or "confusion_matrix" not in self.metrics:
            messagebox.showinfo("Info", "No evaluation metrics available yet!")
            return
        try:
            plot_cm(self.metrics)
        except Exception as e:
            self.log(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_metrics_bar(self):
        if self.metrics is None:
            messagebox.showinfo("Info", "No evaluation metrics available yet!")
            return
                
        model_type = self.model_type.get()
        if model_type == "regression":
            self.log("Plotting regression results...")
            y_test, pred = self.metrics['y_pred'], self.metrics['y_pred']
            plot_regression(y_test, pred)
        elif model_type == "b_classification":
            self.log("Plotting classification visualizations...")
            plot_true_false(self.metrics)  
            
    def save_model(self):
        if self.model is None:
            messagebox.showinfo("Info", "No model to save.")
            return
        try:
            self.model.save()
            self.log("Model saved successfully.")
        except Exception as e:
            self.log(f"Error saving model: {str(e)}")
            messagebox.showerror("Save Error", str(e))

    def load_model(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        model_type = self.model_type.get()
        try:
            if model_type == "regression":
                self.model = RegressionModel()
            elif model_type == "b_classification":
                self.model = BClassifier()
            else:
                raise ValueError("Unknown model type selected.")
            
            self.model.load("weigths/m_classify.joblib")
            self.model_loaded_from_disk = True
            self.log("Model loaded successfully.")
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            messagebox.showerror("Load Error", str(e))



if __name__ == "__main__":
    root = tk.Tk()
    app = FootballModelGUI(root)
    root.mainloop()
