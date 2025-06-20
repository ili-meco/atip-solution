# AI Document Validator Framework

This reusable framework provides an end-to-end solution for document validation with AI capabilities. Originally developed for IMM 5744 consent form processing for ATIP compliance, it has been generalized for use with any document type requiring validation and analysis. The system extracts data from documents using Azure Document Intelligence, processes images (like signatures) with PyMuPDF, and analyzes compliance using Azure OpenAI's GPT-4o model.

## Features
- **Configurable Document Handling**: Process any document type with customizable validation rules.
- **Azure Integration**: Triggers on document uploads to Azure Blob Storage with configurable container names.
- **AI-Powered Analysis**: Uses Azure Document Intelligence with customizable models for data extraction.
- **Image Processing**: Extracts images (like signatures) from documents for further analysis.
- **Customizable Validation**: Flexible validation rules using Azure OpenAI that can be adapted for any compliance requirements.
- **Modern UI**: React frontend with Tailwind CSS for an attractive, responsive user experience.
- **Real-time Feedback**: Live status updates and detailed validation reports with configurable polling intervals.

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
- Verifies that all required signatures are present
- Checks if signatures from both parents are present for minors under 16

## Compliance Rules
The solution validates IMM 5744 consent forms against the following rules:
- All required fields must be filled out.
- Signatures must be in blue ink.
- Date formats must be valid (YYYY-MM-DD).
- Minors under 16 require both parents' signatures or a court order.

## Customizing for Your Use Case

This solution has been designed with reusability in mind. Here's how to customize it for your specific document processing needs:

### Frontend Configuration

Modify the `CONFIG` object in `frontend/src/App.js` to customize:

```javascript
const CONFIG = {
  apiEndpoints: {
    upload: "http://yourapi.com/upload",   // Your backend API endpoint for uploading
    status: "http://yourapi.com/status"    // Your backend API endpoint for status checks
  },
  documentTypes: {
    supportedFormats: [".pdf", ".docx"],   // Add your supported file formats
    acceptString: ".pdf,.docx"             // Update file input accept attribute
  },
  formTitle: "Document Validator",         // Your custom title
  validatorDescription: "Your custom description", // Custom description
  pollingIntervalMs: 2000                  // Status check interval in milliseconds
};
```

### Azure Function Customization

1. **Document Model**: Replace the Document Intelligence model name in `function_app.py` with your trained model.

2. **Processing Logic**: Modify the document processing logic in `function_app.py` to extract the specific data points relevant to your documents.

3. **Validation Rules**: Update the validation rules in the prompt sent to Azure OpenAI to match your document compliance requirements.

### Backend API Customization

1. **Container Names**: Change the blob container names in `backend/main.py` to match your storage structure.

2. **Response Format**: Adjust the response format in the `/status/{filename}` endpoint to include fields specific to your validation requirements.

### Adapting for Different Document Types

To adapt this framework for different document types:

1. Train a custom Document Intelligence model on your specific document type.

2. Update the field extraction logic in `function_app.py`.

3. Modify the Azure OpenAI prompt to include relevant validation criteria for your document type.

4. Update the frontend display logic to show validation results appropriate for your document type.

