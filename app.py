from configs import config
from models import StyleTransferNetwork
from utils.image_utils import imload, imsave

import os
import mimetypes
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import torch
from PIL import Image
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

app = FastAPI()

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path(config.MODEL_PATH)
ckpt = torch.load(str(model_path), map_location=device)
model = StyleTransferNetwork(num_style=config.NUM_STYLE)
model.load_state_dict(ckpt['state_dict'])
model.eval()
model = model.to(device)

# Load the service account credentials
SERVICE_ACCOUNT_FILE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])

# Create the Google Drive API client
drive_service = build('drive', 'v3', credentials=creds)

@app.post("/transfer_style", response_model=dict)
async def transfer_style(content_image: UploadFile = File(...), style_index: int = 0):
    """
    Transfer the style of a given image to a content image.

    Args:
        content_image (UploadFile): The content image file.
        style_index (int): The index of the desired style (0~15 for a specific style, or -1 for all styles). Default is 0.

    Returns:
        The Google Drive file ID of the stylized image.
    """
    # Load the content image and preprocess it
    content_tensor = imload(content_image.file, config.IMSIZE, config.IMSIZE)
    content_tensor = content_tensor.to(device)

    # Generate the style code
    if style_index == -1:
        style_code = torch.eye(config.NUM_STYLE).unsqueeze(-1).to(device)
        content_tensor = content_tensor.repeat(config.NUM_STYLE, 1, 1, 1)
        stylized_image = model(content_tensor, style_code)
    elif style_index in range(config.NUM_STYLE):
        style_code = torch.zeros(1, config.NUM_STYLE, 1, device=device)
        style_code[:, style_index, :] = 1
        stylized_image = model(content_tensor, style_code)
    else:
        raise ValueError("Invalid style index. Should be -1 or between 0 and 15.")

    content_image_extension = Path(content_image.filename).suffix.lstrip('.')
    # If the extension is 'jpg', change it to 'jpeg'
    if content_image_extension.lower() == 'jpg':
        content_image_extension = 'jpeg'

    # Determine the MIME type based on the file extension
    mime_type = mimetypes.guess_type(f'file.{content_image_extension}')[0]

    # Save the stylized image to a buffer
    output_buffer = io.BytesIO()
    imsave(stylized_image, output_buffer, format=content_image_extension)
    output_buffer.seek(0)

    # Construct the stylized image name
    content_image_name = Path(content_image.filename).stem  # Get the file name without extension
    stylized_image_name = f"stylized_{content_image_name}.{content_image_extension}"

    # Upload the stylized image to Google Drive
    folder_id = "1ICcfBIlC2BFh0MpPyT1eROz4afNvqW9i"  # Replace with your Google Drive folder ID
    file_metadata = {'name': stylized_image_name, 'parents': [folder_id]}
    media = MediaIoBaseUpload(output_buffer, mimetype=mime_type, resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    # Get the file ID of the uploaded image
    file_id = file.get('id')

    # Return the Google Drive file ID
    return {"file_id": file_id}

@app.get("/")
def root():
    return {"message": "Style Transfer API is running!"} 