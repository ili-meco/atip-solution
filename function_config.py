"""
Configuration module for the AI Document Validator Azure Function.
Customize these settings to adapt the solution for different document types.
"""
import os

# Azure Blob Storage settings
STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
DOCUMENTS_CONTAINER = "form-pdfs"  # Input documents container
DOCUMENT_DATA_CONTAINER = "form-json"  # Extracted document data
REPORTS_CONTAINER = "form-reports"  # Validation reports
IMAGES_CONTAINER = "form-signatures"  # Extracted images

# Document Intelligence settings
DOCUMENT_INTELLIGENCE_ENDPOINT = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")
DOCUMENT_MODEL_ID = "YourCustomModel"  # Replace with your model ID

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = "gpt-4o"  # Customize as needed

# Validation rules - customize these for your specific document type
VALIDATION_RULES = {
    "requiredFields": [
        "field1",
        "field2",
        "field3"
    ],
    "dateFormatFields": [
        "dateField1",
        "dateField2"
    ],
    "signatureFields": [
        "signature1",
        "signature2"
    ],
    "specialRules": [
        "Custom validation rule 1",
        "Custom validation rule 2"
    ]
}
