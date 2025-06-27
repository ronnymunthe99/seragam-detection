from PIL import Image
import numpy as np
import io
import requests
import time
import os
import gradio as gr  # Correct import - make sure this is gradio library
from detector import ObjectDetectionModel

# Initialize model
detector = ObjectDetectionModel()

try:
    SUPPORTED_LABELS = list(detector.model_seragam.names.values())
except AttributeError:
    SUPPORTED_LABELS = list(detector.model_seragam.names)

# Default images
default_images = {
    "Densus 88": "/home/ronny/object-detection/assets/densus_88.jpeg",
    "Kebaya": "/home/ronny/object-detection/assets/kebaya.jpeg",
    "Paskibra": "/home/ronny/object-detection/assets/paskibra.jpeg"
}

def run_detection(np_image: np.ndarray):
    start_time = time.time()
    
    raw_result = detector.classify_array(np_image)
    cleaned = detector.map_clean_result(raw_result)
    
    elapsed = time.time() - start_time
    
    # Detected labels
    detected_labels = sorted(
        {d.get("labels", "n/a") for d in cleaned.get("cv_attribute", [])}
    )
    
    # Format the output
    output = {
        "processing_time": f"{elapsed:.2f} seconds",
        "detected_labels": ", ".join(detected_labels) if detected_labels else "None detected",
        "full_results": cleaned
    }
    
    return output

def process_image(image_input, input_method):
    if input_method == "Default Image":
        if image_input in default_images:
            image_path = default_images[image_input]
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                return img, run_detection(np.array(img))
            else:
                return None, {"error": f"File not found: {image_path}"}
        else:
            return None, {"error": "Invalid default image selection"}
    
    elif input_method == "Upload Image":
        if image_input is not None:
            img = Image.fromarray(image_input).convert("RGB")
            return img, run_detection(np.array(img))
        else:
            return None, {"error": "No image uploaded"}
    
    elif input_method == "Image URL":
        if image_input:
            try:
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                return img, run_detection(np.array(img))
            except Exception as e:
                return None, {"error": f"Failed to load image: {e}"}
        else:
            return None, {"error": "No URL provided"}

# Create a hidden input component correctly
hidden_input = gr.Textbox(visible=False)

with gr.Blocks(title="Seragam Detection") as demo:
    gr.Markdown("# Seragam Detection")
    
    with gr.Row():
        with gr.Column():
            input_method = gr.Radio(
                label="Select input type",
                choices=["Default Image", "Upload Image", "Image URL"],
                value="Default Image"
            )
            
            # Conditional inputs
            with gr.Group(visible=True) as default_image_group:
                default_image_select = gr.Dropdown(
                    label="Choose a default image",
                    choices=list(default_images.keys()),
                    value=list(default_images.keys())[0]
                )
            
            with gr.Group(visible=False) as upload_image_group:
                image_upload = gr.Image(label="Upload Image")
            
            with gr.Group(visible=False) as url_image_group:
                image_url = gr.Textbox(label="Image URL")
            
            submit_btn = gr.Button("Run Detection", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Processed Image")
            output_result = gr.JSON(label="Detection Results")
    
    # Show/hide input components based on selection
    def toggle_inputs(input_method):
        return [
            gr.Group(visible=input_method == "Default Image"),
            gr.Group(visible=input_method == "Upload Image"),
            gr.Group(visible=input_method == "Image URL")
        ]
    
    input_method.change(
        toggle_inputs,
        inputs=input_method,
        outputs=[default_image_group, upload_image_group, url_image_group]
    )
    
    # Process the image when submit is clicked
    submit_btn.click(
        process_image,
        inputs=[hidden_input, input_method],
        outputs=[output_image, output_result]
    )
    
    # Connect the correct input based on selection
    def get_active_input(input_method, default_img, upload_img, url_img):
        if input_method == "Default Image":
            return default_img
        elif input_method == "Upload Image":
            return upload_img
        elif input_method == "Image URL":
            return url_img
    
    # Update the hidden input when any of the inputs change
    default_image_select.change(
        get_active_input,
        inputs=[input_method, default_image_select, image_upload, image_url],
        outputs=hidden_input
    )
    
    image_upload.change(
        get_active_input,
        inputs=[input_method, default_image_select, image_upload, image_url],
        outputs=hidden_input
    )
    
    image_url.change(
        get_active_input,
        inputs=[input_method, default_image_select, image_upload, image_url],
        outputs=hidden_input
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=1578,
        share=True
    )