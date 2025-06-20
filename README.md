# ATIP Solution: IMM 5744 Consent Form Processing

This project contains an Azure Function to process IMM 5744 consent forms for ATIP compliance. It extracts data from PDFs using Azure Document Intelligence, extracts signature images using PyMuPDF, and analyzes compliance using Azure OpenAI's GPT-4o model.

## Features
- Triggers on PDF uploads to an Azure Blob Storage container (`form-pdfs`).
- Uses a custom Document Intelligence model to extract form data.
- Extracts signature images and analyzes ink color (must be blue) and compliance rules.
- Saves signature snippets as PNG images to `form-signatures` container for visual verification.
- Outputs extracted data to `form-json` and compliance reports to `form-reports`.
- React frontend for validating and submitting PDF forms.
- Real-time validation results display showing issues and recommendations.

## Project Structure
```
atip-general/
├── backend/                 # FastAPI backend service
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/                # React frontend application
│   ├── public/              # Static assets
│   ├── src/                 # React source code
│   └── package.json         # Node dependencies and scripts
├── function_app.py          # Azure Function application
├── host.json                # Azure Function host configuration
├── local.settings.json      # Azure Function local settings (not committed to Git)
└── requirements.txt         # Azure Function dependencies
```

## Prerequisites
- Azure Subscription
- Azure Storage Account with containers: `form-pdfs`, `form-json`, `form-reports`, `form-signatures`
- Azure Document Intelligence resource with a custom model
- Azure OpenAI resource with the `gpt-4o` model enabled
- Python 3.12
- Azure Functions Core Tools (`func`)
- Node.js and npm (for React frontend development)

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/atip-solution.git
   cd atip-solution
   ```

2. **Configure Azure Function**:
   Create a `local.settings.json` file:
   ```json
   {
     "IsEncrypted": false,
     "Values": {
       "AzureWebJobsStorage": "YOUR_STORAGE_CONNECTION_STRING",
       "FUNCTIONS_WORKER_RUNTIME": "python",
       "AZURE_STORAGE_CONNECTION_STRING": "YOUR_STORAGE_CONNECTION_STRING",
       "DOCUMENT_INTELLIGENCE_ENDPOINT": "YOUR_DOCUMENT_INTELLIGENCE_ENDPOINT",
       "DOCUMENT_INTELLIGENCE_KEY": "YOUR_DOCUMENT_INTELLIGENCE_KEY",
       "AZURE_OPENAI_KEY": "YOUR_OPENAI_KEY",
       "AZURE_OPENAI_ENDPOINT": "YOUR_OPENAI_ENDPOINT"
     }
   }
   ```

3. **Install Azure Function Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Backend**:
   Create a `.env` file in the `backend` directory:
   ```
   AZURE_STORAGE_CONNECTION_STRING=YOUR_STORAGE_CONNECTION_STRING
   ```

5. **Install Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

6. **Install Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

7. **Configure Frontend**:
   Update the endpoint URLs in `frontend/src/App.js` if needed (default is `http://localhost:8080`).

## Running the Application

1. **Start the Azure Function**:
   ```bash
   func start
   ```

2. **Start the Backend**:
   ```bash
   cd backend
   uvicorn main:app --reload --port 8080
   ```

3. **Start the Frontend**:
   ```bash
   cd frontend
   npm start
   ```

The application should now be running with:
- Frontend: http://localhost:3000
- Backend: http://localhost:8080
- Azure Function: http://localhost:7071

## Usage
1. Upload a PDF form via the web interface.
2. The backend will store the PDF in Azure Blob Storage.
3. The Azure Function will automatically process the PDF when it's uploaded:
   - Extract form data using Document Intelligence
   - Extract signatures and save as PNGs to the `form-signatures` container
   - Analyze signature ink color and compliance rules
   - Generate a validation report
4. The frontend will poll for validation results and display them when ready.

## Signature Analysis
The solution extracts signatures from the form and:
- Saves each signature as a PNG image in the `form-signatures` container
- Analyzes ink color (blue ink is required for compliance)
  - Detects specific colors (blue, black, red, other) using AI vision capabilities
  - Reports the detected color in the validation results
  - Flags non-blue signatures as issues requiring correction
- Verifies that all required signatures are present
- Checks if signatures from both parents are present for minors under 16

## Compliance Rules
The solution validates IMM 5744 consent forms against the following rules:
- All required fields must be filled out.
- Signatures must be in blue ink.
- Date formats must be valid (YYYY-MM-DD).
- Minors under 16 require both parents' signatures or a court order.

