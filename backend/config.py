"""
Configuration module for the AI Document Validator backend.
Customize these settings to adapt the solution for different document types.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# Container names - customize as needed for your use case
DOCUMENTS_CONTAINER = "form-pdfs"  # Container for uploaded documents
EXTRACTED_DATA_CONTAINER = "form-json"  # Container for extracted data
REPORT_CONTAINER = "form-reports"  # Container for validation reports
IMAGES_CONTAINER = "form-signatures"  # Container for extracted images

# Document validation settings
ALLOWED_FILE_EXTENSIONS = [".pdf"]  # Add more extensions as needed (.docx, etc.)

# API settings
CORS_ORIGINS = ["*"]  # Customize with your frontend origins

# Processing settings
POLLING_INTERVAL_MS = 2000  # Status check polling interval
