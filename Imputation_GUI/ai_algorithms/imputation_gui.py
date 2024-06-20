import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from utils import Dataprepper, Evaluator
from missforest import MissForest


class ImputationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Imputation GUI")

        # Initialize variables
        self.data = None
        self.imputed_data = None
        self.plot_size = 10  # Default number of rows per page
        self.current_page = 1
        self.num_pages = 0
        self.starting_row = 0

        # Create GUI elements
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        self.load_button = tk.Button(top_frame, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(side=tk.LEFT, padx=10)

        algorithm_frame = tk.Frame(top_frame)
        algorithm_frame.pack(side=tk.LEFT)

        tk.Label(algorithm_frame, text="Select Imputation Algorithm:").pack(side=tk.LEFT)

        algorithms = ["MissForest", "KNN"]
        self.selected_algorithm = tk.StringVar(value=algorithms[0])  # Default to MissForest
        self.algorithm_dropdown = tk.OptionMenu(algorithm_frame, self.selected_algorithm, *algorithms)
        self.algorithm_dropdown.pack(side=tk.LEFT, padx=10)

        self.impute_button = tk.Button(top_frame, text="Calculate Missing Data", command=self.impute_and_display)
        self.impute_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(top_frame, text="Save File", command=self.save_imputed_data)
        self.save_button.pack(side=tk.LEFT, padx=10)

        # Frame for displaying results side by side
        results_frame = tk.Frame(root)
        results_frame.pack(pady=10, padx=10)

        # Text widget for displaying original data
        self.original_text = tk.Text(results_frame, wrap=tk.NONE, height=20, width=50, font=("Courier", 10))
        self.original_text.pack(side=tk.LEFT, padx=5)

        # Text widget for displaying imputed data
        self.imputed_text = tk.Text(results_frame, wrap=tk.NONE, height=20, width=50, font=("Courier", 10))
        self.imputed_text.pack(side=tk.LEFT, padx=5)

        # Scrollbars for original data text widget
        self.scrollbar_vertical_original = tk.Scrollbar(results_frame, command=self.original_text.yview)
        self.scrollbar_vertical_original.pack(side=tk.LEFT, fill=tk.Y)
        self.original_text.config(yscrollcommand=self.scrollbar_vertical_original.set)

        self.scrollbar_horizontal_original = tk.Scrollbar(results_frame, command=self.original_text.xview, orient=tk.HORIZONTAL)
        self.scrollbar_horizontal_original.pack(side=tk.TOP, fill=tk.X)
        self.original_text.config(xscrollcommand=self.scrollbar_horizontal_original.set)

        # Scrollbars for imputed data text widget
        self.scrollbar_vertical_imputed = tk.Scrollbar(results_frame, command=self.imputed_text.yview)
        self.scrollbar_vertical_imputed.pack(side=tk.LEFT, fill=tk.Y)
        self.imputed_text.config(yscrollcommand=self.scrollbar_vertical_imputed.set)

        self.scrollbar_horizontal_imputed = tk.Scrollbar(results_frame, command=self.imputed_text.xview, orient=tk.HORIZONTAL)
        self.scrollbar_horizontal_imputed.pack(side=tk.TOP, fill=tk.X)
        self.imputed_text.config(xscrollcommand=self.scrollbar_horizontal_imputed.set)

        # Pagination buttons and settings at the bottom
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=10)

        self.prev_button = tk.Button(bottom_frame, text="Previous Page", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(bottom_frame, text="Next Page", command=self.next_page)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.rows_entry_label = tk.Label(bottom_frame, text="Enter number of rows per page:")
        self.rows_entry_label.pack(side=tk.LEFT, padx=10)

        self.rows_entry = tk.Entry(bottom_frame)
        self.rows_entry.pack(side=tk.LEFT, padx=10)

        self.plot_size_button = tk.Button(bottom_frame, text="Set Plot Size", command=self.set_plot_size)
        self.plot_size_button.pack(side=tk.LEFT, padx=10)

        self.starting_row_label = tk.Label(bottom_frame, text="Starting from row:")
        self.starting_row_label.pack(side=tk.LEFT, padx=10)

        self.starting_row_entry = tk.Entry(bottom_frame)
        self.starting_row_entry.pack(side=tk.LEFT, padx=10)

        self.display_from_button = tk.Button(bottom_frame, text="Display From", command=self.display_from_row)
        self.display_from_button.pack(side=tk.LEFT, padx=10)

        # Label to show current page number
        self.page_label = tk.Label(root, text=f"Page: {self.current_page}/{self.num_pages}")
        self.page_label.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Read the CSV file to check the number of rows
                with open(file_path, 'r') as f:
                    num_rows = sum(1 for line in f)

                if num_rows > 20000:
                    messagebox.showerror("Error",
                                         "Dataset exceeds maximum allowed rows (20000). Please select a smaller dataset.")
                    return

                # Load the dataset into memory
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")

    def impute_and_display(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return

        algorithm = self.selected_algorithm.get()

        try:
            if algorithm == "MissForest":
                imputer = MissForest(n_estimators=100, initial_guess='median', n_iter=5)
                self.imputed_data = pd.DataFrame(imputer.fit_transform(self.data.copy()), columns=self.data.columns)
            elif algorithm == "KNN":
                from sklearn.impute import KNNImputer
                from sklearn.preprocessing import OrdinalEncoder
                import numpy as np

                # Separate numerical and categorical columns
                num_columns = self.data.select_dtypes(include=['number']).columns.tolist()
                cat_columns = self.data.select_dtypes(exclude=['number']).columns.tolist()

                num_data = self.data[num_columns]
                cat_data = self.data[cat_columns].copy()

                # Handle categorical columns with 'M' and 'B'
                for col in cat_columns:
                    if cat_data[col].dtype == 'object':  # Assuming 'M' and 'B' are strings
                        cat_data[col].fillna(cat_data[col].mode().iloc[0], inplace=True)  # Fill NaN with mode

                # Temporarily encode categorical data
                encoder = OrdinalEncoder()
                cat_data_encoded = pd.DataFrame(encoder.fit_transform(cat_data), columns=cat_columns)

                # Combine numerical and encoded categorical data
                prepared_data = pd.concat([num_data, cat_data_encoded], axis=1)

                # Perform KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                imputed_array = imputer.fit_transform(prepared_data)

                # Convert back to DataFrame
                imputed_data = pd.DataFrame(imputed_array, columns=prepared_data.columns)

                # Decode categorical data back to original values
                imputed_data[cat_columns] = encoder.inverse_transform(imputed_data[cat_columns].astype(int))

                # Only replace the missing values in the original data with imputed values
                self.imputed_data = self.data.copy()
                for column in self.data.columns:
                    mask = self.data[column].isna()
                    self.imputed_data.loc[mask, column] = imputed_data.loc[mask, column]
            else:
                messagebox.showerror("Error", "Invalid imputation algorithm selected.")
                return

            # Display results on GUI
            self.update_results_text()

            # Enable page buttons
            self.update_pagination()

        except Exception as e:
            messagebox.showerror("Error", f"Error during imputation: {str(e)}")

    def update_results_text(self):
        if self.data is not None and self.imputed_data is not None:
            self.num_pages = (self.data.shape[0] + self.plot_size - 1) // self.plot_size
            start_idx = (self.current_page - 1) * self.plot_size + self.starting_row
            end_idx = min(start_idx + self.plot_size, self.data.shape[0])

            original_data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
            original_data.insert(0, 'Row Number', range(start_idx + 1, end_idx + 1))
            imputed_data = self.imputed_data.iloc[start_idx:end_idx].reset_index(drop=True)
            imputed_data.insert(0, 'Row Number', range(start_idx + 1, end_idx + 1))

            original_text = original_data.to_string(index=False)
            imputed_text = imputed_data.to_string(index=False)

            self.original_text.delete(1.0, tk.END)  # Clear previous content
            self.original_text.insert(tk.END, original_text)

            self.imputed_text.delete(1.0, tk.END)  # Clear previous content
            self.imputed_text.insert(tk.END, imputed_text)

            # Update page label
            self.page_label.config(text=f"Page: {self.current_page}/{self.num_pages}")
        else:
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(tk.END, "No data to display.")

            self.imputed_text.delete(1.0, tk.END)
            self.imputed_text.insert(tk.END, "No data to display.")

    def set_plot_size(self):
        try:
            plot_size = int(self.rows_entry.get())
            if 1 <= plot_size <= 200:
                self.plot_size = plot_size
                self.current_page = 1  # Reset to first page
                self.update_results_text()
                self.update_pagination()
            else:
                messagebox.showerror("Error", "Please enter a number between 1 and 200.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")

    def display_from_row(self):
        try:
            starting_row = int(self.starting_row_entry.get())
            if 0 < starting_row <= self.data.shape[0]:
                self.starting_row = starting_row - 1
                self.update_results_text()
                self.update_pagination()
            else:
                messagebox.showerror("Error", f"Starting row must be between 1 and {self.data.shape[0]}.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_results_text()
            self.update_pagination()

    def next_page(self):
        if self.current_page < self.num_pages:
            self.current_page += 1
            self.update_results_text()
            self.update_pagination()

    def update_pagination(self):
        self.prev_button.config(state=tk.NORMAL if self.current_page > 1 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_page < self.num_pages else tk.DISABLED)
        self.page_label.config(text=f"Page: {self.current_page}/{self.num_pages}")

    def save_imputed_data(self):
        if self.imputed_data is None:
            messagebox.showerror("Error", "No data to save. Please impute data first.")
            return

        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.imputed_data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data saved successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")

# Main function to run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImputationGUI(root)
    root.mainloop()