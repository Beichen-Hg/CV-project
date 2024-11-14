import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL.Image import Resampling
import cv2

# Import external function modules for detection and classification
from pest_detection import detect_pests
from fruit_classfy import classify_fruit

class FruitPestDetector:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Fruit and Pest Detection System")
        
        # Set window size and background color
        self.root.geometry("800x600")
        self.root.configure(bg='light blue')
        
        # Create and setup the main UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup all UI components"""
        # Create main frame with white background and raised border
        self.main_frame = tk.Frame(
            self.root,
            bg='white',
            bd=2,
            relief=tk.RAISED
        )
        self.main_frame.pack(
            padx=10,
            pady=10,
            fill=tk.BOTH,
            expand=True
        )
        
        # Create image display panel
        self.panel_image = tk.Label(self.main_frame)
        self.panel_image.pack(pady=10)
        
        # Create result display label
        self.label_result = tk.Label(
            self.main_frame,
            text="Please load an image to start detection",
            font=('Arial', 14),
            bg='white'
        )
        self.label_result.pack(pady=10)
        
        # Create load image button
        self.btn_load = tk.Button(
            self.main_frame,
            text="Load Image",
            command=self.load_image,
            font=('Arial', 12),
            bg='#4CAF50',  # Green color
            fg='white',
            padx=20,
            pady=5
        )
        self.btn_load.pack(pady=10)
        
    def load_image(self):
        """Handle image loading and processing"""
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")
            ]
        )
        
        if file_path:
            try:
                # Load and resize image for display
                img = Image.open(file_path)
                img = img.resize((400, 400), Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                # Update image display
                self.panel_image.config(image=img_tk)
                self.panel_image.image = img_tk
                
                # Perform detection and classification
                pest_result = detect_pests(file_path)
                fruit_result = classify_fruit(file_path)
                
                # Update result display
                result_text = f"Pest Detection: {pest_result}\nFruit Type: {fruit_result}"
                self.label_result.config(
                    text=result_text,
                    fg='#333333'  # Dark gray text color
                )
                
            except Exception as e:
                # Handle potential errors
                self.label_result.config(
                    text=f"Error: {str(e)}",
                    fg='red'
                )
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Create and run the application
if __name__ == "__main__":
    app = FruitPestDetector()
    app.run()
