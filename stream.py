import streamlit as st
import requests
from configs.config import FASTAPI_URI
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import os
# Define the URI of the FastAPI server

# Function to upload and stylize image
def transfer_style(content_image, style_index):
    files = {"content_image": content_image}
    params = {"style_index": style_index}
    response = requests.post(f"{FASTAPI_URI}/transfer_style", files=files, params=params)
    if response.status_code == 200:
        json_data = response.json()
        if "file_id" in json_data:
            return json_data["file_id"]
        else:
            st.error("Failed to retrieve stylized image file ID from the server.")
    else:
        st.error(f"Failed to perform style transfer. Server returned status code: {response.status_code}")
        try:
            error_message = response.json()["detail"]
            st.error(f"Error message: {error_message}")
        except (KeyError, ValueError):
            st.error("No error message available from the server.")

def download_image_from_drive(file_id, creds):
    drive_service = build('drive', 'v3', credentials=creds)
    request = drive_service.files().get_media(fileId=file_id)
    
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    file_buffer.seek(0)
    return file_buffer

# Streamlit app
def main():
    st.title("Neural Style Transfer")

    # Introduction
    st.write("A PyTorch Implementation")

    # Display the image and label
    st.image("./imgs/styles.jpg", caption="Styles Available", use_column_width=True)

    # File uploader for content image
    st.subheader("Upload Content Image")
    content_image = st.file_uploader("Choose a content image", type=["jpg", "jpeg", "png"])

    if content_image is not None:
        style_index = st.slider("Select Style Index (0-15, -1 for all styles)", min_value=-1, max_value=15, value=0)
        if style_index != -1:
            st.warning("Note: Stylizing with a specific style index.")
        else:
            st.warning("Note: Stylizing with all styles.")

        if st.button("Transfer Style"):
            if style_index == -1 or (style_index >= 0 and style_index <= 15):
                with st.spinner("Stylizing..."):
                    stylized_image_id = transfer_style(content_image, style_index)
                    if stylized_image_id:
                        # Load the service account credentials
                        SERVICE_ACCOUNT_FILE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])

                        # Download the stylized image from Google Drive
                        stylized_image_buffer = download_image_from_drive(stylized_image_id, creds)

                        # Display the stylized image in Streamlit
                        st.image(stylized_image_buffer, caption="Stylized Image", use_column_width=True)
            else:
                st.error("Invalid style index. Please select a value between 0 and 15 or -1.")

if __name__ == "__main__":
    main()