import ipyleaflet as L
from faicons import icon_svg
from geopy.distance import geodesic, great_circle
from ipywidgets import FileUpload
from pytest import Session  # Import FileUpload widget for image upload
from shared import BASEMAPS, CITIES
from shiny import Inputs, Outputs, reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget
from shiny.types import ImgData, FileInfo
from PIL import Image 
import io
from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


# city_names = sorted(list(CITIES.keys()))
# Load the model configuration

ui.page_opts(title="Sidewalk Detector", fillable=True)
{"class": "bslib-page-dashboard"}
ui.input_dark_mode(mode="dark")


with ui.layout_column_wrap(fill=False):
    ui.input_file("input_image", "Choose TIFF File", accept=[".tif", ".tiff"], multiple=False)

#render input image
with ui.card():
    ui.card_header("Image Uploaded")

    @render.image
    def image():
        file: list[FileInfo] | None = input.input_image()
        
        # Perform TIFF to PNG conversion
        
        if file:
            tmp_save_file = get_temp_save_file(file[0]["datapath"], "png")
            success = convert_tiff_to_png_in_memory(file[0]["datapath"], tmp_save_file)
            if success:
                img: ImgData = {"src": tmp_save_file, "style": {"max-width": "100%"}}
                return img
            else:
                return None
        else:
            return None

#render mask
with ui.layout_columns():
    with ui.card():
        ui.card_header("Generated Prediction Mask")

        @render.image
        def mask_pred():
            file: list[FileInfo] | None = input.input_image()
            if file:
                tmp_prob_file = get_temp_save_file(file[0]["datapath"], "prob")
                tmp_pred_file = get_temp_save_file(file[0]["datapath"], "pred")
                success = get_mask(file[0]["datapath"], tmp_prob_file, tmp_pred_file)
                if success:
                    img: ImgData = {"src": tmp_prob_file, "style": {"max-width": "100%"}}
                    return img
                else:
                    return None
            else:
                return None
    
    with ui.card():
        ui.card_header("Generated Probability Map")

        @render.image
        def mask_prob():
            file: list[FileInfo] | None = input.input_image()
            if file:
                tmp_pred_file = get_temp_save_file(file[0]["datapath"], "pred")
                img: ImgData = {"src": tmp_pred_file, "style": {"max-width": "100%"}}
                return img
            else:
                return None


def get_temp_save_file(tiff_file, usage: str):
    from pathlib import Path
    upload_loc = Path(tiff_file)
    tmp_folder = str(upload_loc.parent)
    tif_name = str(upload_loc.name)
    no_ext_name = tif_name.strip(".tiff") if tif_name.endswith(".tiff") else tif_name.strip(".tif")
    if usage == "png":
        png_name = no_ext_name + ".png"
        tmp_save_file = tmp_folder + f"\\{png_name}"
        return tmp_save_file
    elif usage == "prob":
        mask_name = no_ext_name + "prob.png"
        tmp_save_file = tmp_folder + f"\\{mask_name}"
        return tmp_save_file
    elif usage == "pred":
        mask_name = no_ext_name + "pred.png"
        tmp_save_file = tmp_folder + f"\\{mask_name}"
        return tmp_save_file
    else:
        return None

def convert_tiff_to_png_in_memory(tiff_file, tmp_save_file):
    try:
        # Open the TIFF image from bytes data
        with Image.open(tiff_file) as img:
            # Create an in-memory PNG image
            png_data = io.BytesIO()
            
            img.save(png_data, format="PNG")
            png_data.seek(0)  # Reset the stream position
        
            with open(tmp_save_file, "wb") as png_file:
                png_file.write(png_data.getvalue())
        
        return True
    except Exception as e:
        print(f"Error converting TIFF to PNG in memory: {e}")
        return False

def get_mask(tiff_file, tmp_prob_file, tmp_pred_file):
    try:
        init_sam_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        global sam_model, processor
        input_image = Image.open(tiff_file)
        inputs = processor(input_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sam_model.eval()

        with torch.no_grad():
            outputs = sam_model(**inputs, multimask_output=False)

        # apply sigmoid
        image_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        
        # convert soft mask to hard mask
        image_prob = image_prob.cpu().numpy().squeeze()
        image_prediction = (image_prob > 0.5).astype(np.uint8)

        write_img_to_file(tmp_prob_file, image_prob)
        write_img_to_file(tmp_pred_file, image_prediction, 'PRGn')
        return True
    except Exception as e:
        print(f"Error in generating masks: {str(e)}")
        return False



processor = None
sam_model = None

def init_sam_model():
    global sam_model, processor
    
    # Create an instance of the model architecture with the loaded configuration
    if not sam_model:
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        sam_model = SamModel(config=model_config)
        #Update the model by loading the weights from saved file.
        sam_model.load_state_dict(torch.load("SAM_model_cp/sidewalks_model_checkpoint_full_train.pth"))
        # set the device to cuda if available, otherwise use cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_model.to(device)

def write_img_to_file(filepath, image_array, color = None):
    if color:
        # Create a color map (e.g., using 'viridis' colormap for segmentation)
        cmap = plt.get_cmap(color)

        # Normalize the image_array values to be between 0 and 1
        normalized_image = image_array.astype(np.float32) / np.max(image_array)

        # Apply the colormap to the normalized image to get RGBA values
        rgba_image = cmap(normalized_image)[:, :, :3]  # Ignore alpha channel (4th channel)

        # Convert to uint8 (0-255 range) for PIL image
        rgba_image_uint8 = (rgba_image * 255).astype(np.uint8)

        # Create PIL Image object from the RGBA NumPy array
        image = Image.fromarray(rgba_image_uint8)
    else:
        image_array = (image_array * 255).astype(np.uint8) 

        image = Image.fromarray(image_array)
    # Save the image to a PNG file
    image.save(filepath)