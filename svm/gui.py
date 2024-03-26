import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import lemna_master
import os
import glob
from PIL import Image, ImageTk
import threading


def get_first_model_path(directory, extension=".sav"):
    """Search for the first model file with the specified extension in the given directory."""
    model_files = glob.glob(os.path.join(directory, f"*{extension}"))
    return model_files[0] if model_files else None

# GUI application
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Lemna-Master5000")

        self.lemna_master = lemna_master.LemnaMaster()

        # Search for .sav files in the current directory and use the first one found
        self.current_directory = os.getcwd()
        self.model_path = get_first_model_path(self.current_directory, ".sav")
        self.lemna_master.load_model(self.model_path)

        # Button to select a different model
        self.change_model_btn = tk.Button(root, text="Select Model", command=self.change_model)
        self.change_model_btn.pack(pady=10)

        model_display_text = f"Current Model: {os.path.basename(self.model_path)}" if self.model_path else "No model found"
        self.model_label = tk.Label(root, text=model_display_text)
        self.model_label.pack(pady=5)

        # Training button and label
        self.train_btn = tk.Button(root, text="Train new Model", command=self.train_model)
        self.train_btn.pack(pady=10)

        # Initialize the BooleanVar to hold the toggle state
        self.toggle_state = tk.BooleanVar(value=False)
        # Toggle switch (Checkbutton) for extra marking
        self.toggle_btn = tk.Checkbutton(root, text="Tote Wasserlinsen extra markieren?", variable=self.toggle_state,
                                         onvalue=True, offvalue=False)
        self.toggle_btn.pack(pady=10)

        # Inference button, label, and progress bar
        self.infer_btn = tk.Button(root, text="Run Inference", command=self.run_inference)
        self.infer_btn.pack(pady=10)
        self.infer_folder_label = tk.Label(root, text="No folder selected for inference")
        self.infer_folder_label.pack(pady=5)
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(pady=10)

        self.inference_cancelled = False
        self.cancel_button = tk.Button(self.root, text="Cancel Inference", command=self.request_stop_inference)
        self.cancel_button.pack()  # Adjust the layout according to your GUI design

        self.plant_area_label = tk.Label(root, text="")
        self.plant_area_label.pack(pady=5, padx=20)
        # Image display frames
        self.frame = tk.Frame(root)
        self.frame.pack(padx=5, pady=5)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack()

    def request_stop_inference(self):
        self.inference_cancelled = True

    def display_image(self, image):
        """Display images in the application."""

        image = Image.fromarray(image.astype('uint8'))  # Convert NumPy array to PIL Image
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference

    def change_model(self):
        """Open a dialog for the user to select a new model file."""
        new_model_path = filedialog.askopenfilename(initialdir=self.current_directory, title="Select Model File, must end with '.sav'",
                                                    filetypes=[("Model files", "*.sav")])
        if new_model_path:
            self.model_path = new_model_path
            self.lemna_master.load_model(self.model_path)
            self.model_label.config(text=f"Current Model: {os.path.basename(self.model_path)}")
            messagebox.showinfo("Model Changed", f"Model changed to {os.path.basename(self.model_path)}.")


    def train_model(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            #self.train_folder_label.config(text=f"Training folder: {folder_path}")

            model, path, accuracy = self.lemna_master.train_svm(folder_path, self.toggle_state.get())
            self.model = model  # Update the current model
            self.model_label.config(text=f"Current Model: {path}")
            messagebox.showinfo("Training Complete\n", f"The model has been trained and saved successfully. \n"
                                                       f"Overall Accuracy: {round(accuracy*100, 2)}%")
        else:
            messagebox.showwarning("Training Aborted", "No folder selected for training.")

    """def run_inference(self):
        if not self.model:
            messagebox.showwarning("Inference Error", "Please load or train a model before running inference.")
            return
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.infer_folder_label.config(text=f"Inference folder: {folder_path}")
            self.progress['value'] = 0
            self.root.update_idletasks()

            files_to_test = self.lemna_master.read_all_images_in_folder(folder_path)
            self.root.config(cursor="watch")
            self.root.update()
            for i, image_path in enumerate(files_to_test):
                print(f"Processing {image_path}")
                output_image, plant_percentage = self.lemna_master.run_inference_on_image(os.path.join(folder_path, image_path))
                self.plant_area_label["text"] = f"Pflanzenfläche: {plant_percentage} %"
                self.display_image(output_image)
                # Update progress in GUI
                progress = int((i + 1) / len(files_to_test) * 100)
                app.update_progress(progress)
            self.root.config(cursor="")
            messagebox.showinfo("Inference Complete", "Inference has been run successfully.")
        else:
            messagebox.showwarning("Inference Aborted", "No folder selected for inference.")"""

    def run_inference(self):
        if not self.lemna_master.loaded_model:
            self.lemna_master.load_model(self.model_path)
        inference_folder_path = filedialog.askdirectory()
        if inference_folder_path:
            self.infer_folder_label.config(text=f"Inference folder: {inference_folder_path}")
            self.progress['value'] = 0
            self.root.update_idletasks()

            self.root.config(cursor="watch")
            threading.Thread(target=self.inference_thread, args=(inference_folder_path,)).start()
            self.root.config(cursor="")

    def inference_thread(self, folder_path):
        self.inference_cancelled = False  # Reset at the start
        files_to_test = self.lemna_master.read_all_images_in_folder(folder_path)
        for i, image_path in enumerate(files_to_test):
            if self.inference_cancelled:
                break  # Stop processing if cancellation is requested
            print(f"Processing {image_path}")
            output_image, plant_percentage = self.lemna_master.run_inference_on_image(
                os.path.join(folder_path, image_path))
            self.update_gui_after_inference(output_image, plant_percentage, i, len(files_to_test))

        if self.inference_cancelled:
            self.root.after(0, lambda: messagebox.showinfo("Inference Cancelled", "Inference has been cancelled."))
        else:
            self.root.after(0,
                            lambda: messagebox.showinfo("Inference Complete", "Inference has been run successfully."))

        self.inference_cancelled = False  # Reset at the end

    def update_gui_after_inference(self, output_image, plant_percentage, i, total):
        def update():
            self.plant_area_label["text"] = f"Pflanzenfläche: {plant_percentage} %"
            self.display_image(output_image)
            progress = int((i + 1) / total * 100)
            self.update_progress(progress)

        self.root.after(0, update)

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()

    root.geometry("800x600")
    app = AppGUI(root)
    root.mainloop()