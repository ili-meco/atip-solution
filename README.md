# ATIP Solution: IMM 5744 Consent Form Processing

This project contains an Azure Function to process IMM 5744 consent forms for ATIP compliance. It extracts data from PDFs using Azure Document Intelligence, extracts signature images using PyMuPDF, and analyzes compliance using Azure OpenAI's GPT-4o model.

## Features
- Triggers on PDF uploads to an Azure Blob Storage container (`form-pdfs`).
- Uses a custom Document Intelligence model (`IMM5744E-ConsentForm-3v2`) to extract form data.
- Extracts signature images and analyzes ink color (must be blue) and compliance rules.
- Outputs extracted data to `form-json` and compliance reports to `form-reports`.

## Prerequisites
- Azure Subscription
- Azure Storage Account with containers: `form-pdfs`, `form-json`, `form-reports`
- Azure Document Intelligence resource with a custom model named `IMM5744E-ConsentForm-3v2`
- Azure OpenAI resource with the `gpt-4o` model enabled
- Python 3.12
- Azure Functions Core Tools (`func`)

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/atip-solution.git
   cd atip-solution