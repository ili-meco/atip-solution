import azure.functions as func
import json
import logging
import base64
import fitz  # PyMuPDF
from typing import Dict, List, Optional
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, BoundingRegion, DocumentSpan
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from datetime import datetime
import uuid
import re
import os

app = func.FunctionApp()

# Configuration from environment variables
DI_ENDPOINT = os.environ["DI_ENDPOINT"]
DI_KEY = os.environ["DI_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
STORAGE_CONNECTION_STRING = os.environ["atipblobfunction_STORAGE"]
MODEL_ID = "IMM5744E-ConsentForm-3v2"

def calculate_age(dob: str) -> Optional[int]:
    """Calculate age based on date of birth (YYYY-MM-DD)."""
    try:
        dob_date = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        logging.info(f"Calculated age for DOB {dob}: {age} years")
        return age
    except ValueError as e:
        logging.warning(f"Invalid DOB format for {dob}: {str(e)}")
        return None

def extract_signature_image(pdf_data: bytes, page_num: int, bounding_box: list, field_name: str) -> Optional[str]:
    """Extract a signature image from a PDF using PyMuPDF."""
    try:
        # Validate bounding_box: must be a list of at least 8 floats (quadrilateral)
        if not isinstance(bounding_box, list) or len(bounding_box) < 8 or not all(isinstance(x, float) for x in bounding_box):
            logging.warning(f"Invalid bounding box for field '{field_name}' on page {page_num}: {bounding_box}. Skipping image extraction.")
            return None

        # Validate that coordinates are paired (x, y)
        coordinates = [(bounding_box[i], bounding_box[i+1]) for i in range(0, len(bounding_box), 2)]
        if len(coordinates) < 4:
            logging.warning(f"Insufficient coordinate pairs for field '{field_name}' on page {page_num}: {coordinates}. Skipping image extraction.")
            return None

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        page = pdf_document[page_num - 1]
        rect = fitz.Rect(
            min(p[0] for p in coordinates),
            min(p[1] for p in coordinates),
            max(p[0] for p in coordinates),
            max(p[1] for p in coordinates)
        )
        # Increase resolution to 600 DPI for better image clarity
        pix = page.get_pixmap(matrix=fitz.Matrix(600/72, 600/72), clip=rect)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        logging.info(f"Extracted signature image for field '{field_name}' on page {page_num}. Image size: {len(img_bytes)} bytes")
        pdf_document.close()
        return img_base64
    except Exception as e:
        logging.error(f"Failed to extract signature image for field '{field_name}' on page {page_num}: {str(e)}")
        return None

def validate_form_fields(fields: Dict, doc_result: AnalyzeResult) -> List[Dict]:
    """Validate required fields and signatures in the extracted form data using DI results."""
    issues = []
    
    # Required fields for Designated Representative (Section 1)
    rep_fields = [
        "1. Designated Representative's Information Family name (surname)",
        "1. Designated Representative's Information Given name(s)",
        "1. Designated Representative's Information Firm/organization"
    ]
    for field in rep_fields:
        field_value = fields.get(field, {}).get("valueString")
        logging.info(f"Validating Section 1 field '{field}': valueString={field_value}")
        if field not in fields or field_value is None:
            issues.append({
                "description": f"Missing or empty field in 1. Designated Representative’s Information: {field.split('Information ')[-1]}",
                "field": field,
                "page": fields[field].get("boundingRegions", [{}])[0].get("pageNumber", 1) if field in fields else 1,
                "action": "Request missing information"
            })

    # Check Applicant's Information (Section 2)
    applicant_fields = [
        "2. Applicant's Information Family name (surname)",
        "2. Applicant's Information Given name(s)",
        "2. Applicant's Information Date of birth (YYYY-MM-DD)"
    ]
    for field in applicant_fields:
        field_value = fields.get(field, {}).get("valueString")
        logging.info(f"Validating Section 2 field '{field}': valueString={field_value}")
        if field not in fields or field_value is None:
            issues.append({
                "description": f"Missing or empty field in 2. Applicant’s Information: {field.split('Information ')[-1]}",
                "field": field,
                "page": fields[field].get("boundingRegions", [{}])[0].get("pageNumber", 1) if field in fields else 1,
                "action": "Request missing information"
            })

    # Validate date of birth format (YYYY-MM-DD) for Applicant
    dob_field = "2. Applicant's Information Date of birth (YYYY-MM-DD)"
    dob_value = fields.get(dob_field, {}).get("valueString")
    logging.info(f"Validating Applicant DOB field '{dob_field}': valueString={dob_value}")
    if dob_value is not None:
        try:
            datetime.strptime(dob_value, "%Y-%m-%d")
        except ValueError:
            issues.append({
                "description": f"Invalid date format in 2. Applicant’s Information: Date of birth ({dob_value})",
                "field": dob_field,
                "page": fields[dob_field].get("boundingRegions", [{}])[0].get("pageNumber", 1),
                "action": "Request valid date format"
            })

    # Check Applicant's signature
    applicant_signature = "2. Applicant's Information Signature"
    applicant_signature_value = fields.get(applicant_signature, {}).get("valueSignature")
    logging.info(f"Validating Applicant signature '{applicant_signature}': valueSignature={applicant_signature_value}")
    if applicant_signature not in fields or applicant_signature_value != "signed":
        issues.append({
            "description": "Missing signature in 2. Applicant’s Information",
            "field": applicant_signature,
            "page": fields[applicant_signature].get("boundingRegions", [{}])[0].get("pageNumber", 1) if applicant_signature in fields else 1,
            "action": "Request signature"
        })

    # Check Related Individuals (2.1, 2.2, 2.3)
    for idx, section in enumerate(["2.1", "2.2", "2.3"]):
        # Define fields for the section
        family_name = f"{section} Related Individual's Information Family name (surname)"
        given_name = f"{section} Related Individual's Information Given name(s)"
        dob = f"{section} Related Individual's Information Date of birth (YYYY-MM-DD)"
        signature = f"{section} Related Individual's Information Signature"

        # Validate each field independently
        required_fields = [family_name, given_name, dob]
        for field in required_fields:
            field_value = fields.get(field, {}).get("valueString")
            logging.info(f"Validating Section {section} field '{field}': valueString={field_value}")
            if field not in fields or field_value is None:
                issues.append({
                    "description": f"Missing or empty field in {section} Related Individual’s Information: {field.split('Information ')[-1]}",
                    "field": field,
                    "page": fields[field].get("boundingRegions", [{}])[0].get("pageNumber", 1) if field in fields else 1,
                    "action": "Request missing information"
                })

        # Validate DOB format if present
        dob_value = fields.get(dob, {}).get("valueString")
        if dob_value is not None:
            try:
                datetime.strptime(dob_value, "%Y-%m-%d")
            except ValueError:
                issues.append({
                    "description": f"Invalid date format in {section} Related Individual’s Information: Date of birth ({dob_value})",
                    "field": dob,
                    "page": fields[dob].get("boundingRegions", [{}])[0].get("pageNumber", 1),
                    "action": "Request valid date format"
                })

        # Validate signature independently
        signature_value = fields.get(signature, {}).get("valueSignature")
        logging.info(f"Validating Section {section} signature '{signature}': valueSignature={signature_value}")
        if signature not in fields or signature_value != "signed":
            issues.append({
                "description": f"Missing signature in {section} Related Individual’s Information",
                "field": signature,
                "page": fields[signature].get("boundingRegions", [{}])[0].get("pageNumber", 1) if signature in fields else 1,
                "action": "Request signature"
            })

    # Validate consent validity date
    consent_date_str = None
    for para in doc_result.paragraphs or []:
        if para.bounding_regions and para.bounding_regions[0].page_number == 1:
            if "Date (YYYY-MM-DD)" in para.content and para.content.startswith("202"):
                consent_date_str = para.content.split()[0]
                break
    if consent_date_str:
        logging.info(f"Validating consent date: {consent_date_str}")
        try:
            consent_date = datetime.strptime(consent_date_str, "%Y-%m-%d")
            if (datetime.now() - consent_date).days > 365:
                issues.append({
                    "description": "Consent is expired (valid for one year from signature date)",
                    "field": "Consent Date",
                    "page": 1,
                    "action": "Request updated consent"
                })
        except ValueError:
            issues.append({
                "description": f"Invalid consent date format: {consent_date_str}",
                "field": "Consent Date",
                "page": 1,
                "action": "Request valid date format"
            })

    return issues

def analyze_signatures_with_gpt4o(fields: Dict, doc_result: AnalyzeResult, pdf_data: bytes, blob_name: str) -> tuple:
    """Use Azure OpenAI GPT-4o to analyze signatures for ink color and compliance."""
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-01"
    )
    
    # Extract structured data for GPT-4o
    extracted_data = {
        "form_metadata": {"form_version": "IMM 5744 (09-2018) E", "submission_date": "Unknown"},
        "applicant": {},
        "related_individuals": [{} for _ in range(3)]
    }
    signature_images = []
    
    # Applicant data
    applicant_fields = [
        "2. Applicant's Information Family name (surname)",
        "2. Applicant's Information Given name(s)",
        "2. Applicant's Information Date of birth (YYYY-MM-DD)",
        "2. Applicant's Information Signature"
    ]
    for field in applicant_fields:
        if field in fields:
            key = field.split("Information ")[-1].lower().replace(" ", "_")
            if key == "date_of_birth_(yyyy-mm-dd)":
                value = fields[field].get("valueString", "")
                age = calculate_age(value)
                extracted_data["applicant"]["dob"] = {"value": value, "age": age}
            elif key == "signature":
                is_present = fields[field].get("valueSignature") == "signed"
                bounding_region = fields[field].get("boundingRegions", [{}])[0]
                page = bounding_region.get("pageNumber", 1)
                bounding_box = bounding_region.get("polygon", [])
                extracted_data["applicant"]["signature"] = {
                    "present": is_present,
                    "page": page,
                    "bounding_box": bounding_box
                }
                if is_present and isinstance(bounding_box, list) and len(bounding_box) >= 8:
                    img_base64 = extract_signature_image(pdf_data, page, bounding_box, field)
                    if img_base64:
                        signature_images.append({"section": "applicant", "image": img_base64, "page": page})
                    else:
                        logging.warning(f"No valid signature image extracted for field '{field}' on page {page}")
                        extracted_data["applicant"]["signature"]["is_blue"] = None
                else:
                    logging.warning(f"Invalid or missing bounding box for field '{field}' on page {page}: {bounding_box}")
                    extracted_data["applicant"]["signature"]["is_blue"] = None
            else:
                extracted_data["applicant"][key] = {"value": fields[field].get("valueString", "")}
    
    # Related individuals data
    for i, section in enumerate(["2.1", "2.2", "2.3"]):
        related_fields = [
            f"{section} Related Individual's Information Family name (surname)",
            f"{section} Related Individual's Information Given name(s)",
            f"{section} Related Individual's Information Date of birth (YYYY-MM-DD)",
            f"{section} Related Individual's Information Signature"
        ]
        for field in related_fields:
            if field in fields:
                key = field.split("Information ")[-1].lower().replace(" ", "_")
                if key == "date_of_birth_(yyyy-mm-dd)":
                    value = fields[field].get("valueString", "")
                    age = calculate_age(value)
                    extracted_data["related_individuals"][i]["dob"] = {"value": value, "age": age}
                elif key == "signature":
                    is_present = fields[field].get("valueSignature") == "signed"
                    bounding_region = fields[field].get("boundingRegions", [{}])[0]
                    page = bounding_region.get("pageNumber", 1)
                    bounding_box = bounding_region.get("polygon", [])
                    extracted_data["related_individuals"][i]["signature"] = {
                        "present": is_present,
                        "page": page,
                        "bounding_box": bounding_box
                    }
                    if is_present and isinstance(bounding_box, list) and len(bounding_box) >= 8:
                        img_base64 = extract_signature_image(pdf_data, page, bounding_box, field)
                        if img_base64:
                            signature_images.append({"section": f"related_individual_{i}", "image": img_base64, "page": page})
                        else:
                            logging.warning(f"No valid signature image extracted for field '{field}' on page {page}")
                            extracted_data["related_individuals"][i]["signature"]["is_blue"] = None
                    else:
                        logging.warning(f"Invalid or missing bounding box for field '{field}' on page {page}: {bounding_box}")
                        extracted_data["related_individuals"][i]["signature"]["is_blue"] = None
                else:
                    extracted_data["related_individuals"][i][key] = {"value": fields[field].get("valueString", "")}
    
    # GPT-4o prompt
    prompt = f"""
    You are an expert in analyzing the IMM 5744 E consent form for ATIP compliance. Perform these tasks based solely on the provided input data and images, without making assumptions or inferring missing information:
    **ATIP Consent Form Instructions**:
    - The form has three main sections:
      - Section 1: Designated Representative’s Information (requester details).
      - Section 2: Applicant’s Information.
      - Sections 2.1 to 2.3: Related Individual’s Information (up to 3 related individuals).
    - Minors under 16 require both parents’ signatures in any shade of blue, or a court order.
    - Signatures must be original, in any shade of blue (e.g., navy, light blue, turquoise), and include the date. No other colors (e.g., black, red, green) are allowed. No electronic signatures.
    - Required fields (family_name, given_name, dob) must be present and non-empty for any section with data.
    **Tasks**:
    1. **Ink Color Detection**:
       - For each signature field, if 'present' is false or no valid image is provided, mark as missing ('is_blue': null).
       - If 'present' is true and an image is provided, analyze the base64-encoded PNG image to determine if the ink is any shade of blue (e.g., navy, light blue, turquoise). Set 'is_blue': true if blue, false if not. Be cautious to avoid misidentifying blue ink as another color.
       - Include page number and section (e.g., 'applicant', 'related_individual_0').
    2. **Compliance Check**:
       - Verify these rules:
         - Individuals 16+ must have their own signature in any shade of blue.
         - Minors (<16) must have signatures from both parents or a court order, in any shade of blue.
         - Signatures must be in blue ink and include a date.
         - Required fields (family_name, given_name, dob) must be present and non-empty for any section with data.
       - For each issue, specify the page number, section name (e.g., '2. Applicant's Information'), and individual (e.g., 'Applicant Jason Jenna (age 1)').
       - Provide clear action recommendations (e.g., 'Request missing signature', 'Verify ink color manually').
       - Do not report ink color issues for missing signatures or signatures without valid images.
       - Do not report issues about exceeding the limit of related individuals; focus only on field inputs and signatures.
    **Input Data**:
    {json.dumps(extracted_data, indent=2)}
    **Signature Images** (base64-encoded PNGs):
    """
    for sig in signature_images:
        prompt += f"\n\nSection: {sig['section']}, Page: {sig['page']}\n![Signature](data:image/png;base64,{sig['image']})\n"
    prompt += f"""
    **Output a JSON object with**:
    - 'form_id': '{blob_name}'
    - 'signatures': List of objects, each with 'section' (e.g., 'applicant'), 'is_blue' (boolean or null), 'page' (integer).
    - 'issues': List of objects with 'description' (specific, e.g., 'Signature not in blue ink for Applicant Jason Jenna (age 1) in 2. Applicant's Information'), 'field' (e.g., 'Signature'), 'page' (integer), and 'action' (e.g., 'Request signature in blue ink'). Exclude any issues about exceeding related individuals limit.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        result = json.loads(response.choices[0].message.content)
        logging.info(f"GPT-4o response: {json.dumps(result, indent=2)}")
        
        # Filter out issues related to exceeding related individuals limit
        filtered_issues = [
            issue for issue in result.get("issues", [])
            if "exceeded limit of related individuals" not in issue.get("description", "").lower()
        ]
        result["issues"] = filtered_issues
        return result.get("signatures", []), filtered_issues
    except Exception as e:
        logging.error(f"Error analyzing signatures with GPT-4o: {str(e)}")
        return [], [{"description": f"Failed to analyze signatures: {str(e)}", "field": "Signature Analysis", "page": 1, "action": "Flag for review"}]

def serialize_bounding_region(region: Optional[BoundingRegion]) -> Optional[Dict]:
    """Serialize a BoundingRegion object to a JSON-compatible dictionary."""
    if not region:
        return None
    return {
        "pageNumber": region.page_number,
        "polygon": region.polygon
    }

def serialize_document_span(span: Optional[DocumentSpan]) -> Optional[Dict]:
    """Serialize a DocumentSpan object to a JSON-compatible dictionary."""
    if not span:
        return None
    return {
        "offset": span.offset,
        "length": span.length
    }

def serialize_field(field) -> Optional[Dict]:
    """Serialize a document field to a JSON-compatible dictionary."""
    if not field:
        return None
    return {
        "valueString": field.get("valueString"),
        "valueSignature": field.get("valueSignature"),
        "boundingRegions": [serialize_bounding_region(r) for r in field.get("boundingRegions", [])],
        "content": field.get("content")
    }

def serialize_analyze_result(result: AnalyzeResult) -> Dict:
    """Manually serialize AnalyzeResult to a dictionary for JSON storage."""
    serialized = {
        "apiVersion": result.api_version,
        "modelId": result.model_id,
        "documents": [
            {
                "docType": doc.doc_type,
                "fields": {k: serialize_field(v) for k, v in (doc.fields or {}).items()},
                "boundingRegions": [serialize_bounding_region(r) for r in doc.bounding_regions or []],
                "spans": [serialize_document_span(s) for s in doc.spans or []]
            } for doc in result.documents or []
        ],
        "pages": [
            {
                "pageNumber": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "spans": [serialize_document_span(s) for s in page.spans or []]
            } for page in result.pages or []
        ],
        "paragraphs": [
            {
                "role": para.role,
                "content": para.content,
                "boundingRegions": [serialize_bounding_region(r) for r in para.bounding_regions or []],
                "spans": [serialize_document_span(s) for s in para.spans or []]
            } for para in result.paragraphs or []
        ]
    }
    return serialized

def save_json_to_blob(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, data: Dict):
    """Save JSON data to a blob in the specified container."""
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True)
    logging.info(f"Saved JSON to {container_name}/{blob_name}")

@app.function_name(name="ProcessATIPForm")
@app.blob_trigger(arg_name="inputBlob", path="form-pdfs/{name}.pdf", connection="AzureWebJobsStorage")
@app.blob_output(arg_name="jsonOutput", path="form-json/{name}.json", connection="AzureWebJobsStorage")
@app.blob_output(arg_name="reportOutput", path="form-reports/{name}_report.json", connection="AzureWebJobsStorage")
def process_atip_form(inputBlob: func.InputStream, jsonOutput: func.Out[str], reportOutput: func.Out[str]):
    """Azure Function to process IMM 5744E Consent Form and generate a recommendation report."""
    logging.info(f"Processing blob: {inputBlob.name}")
    
    try:
        # Initialize Blob Service Client
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

        # Initialize Document Intelligence client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=DI_ENDPOINT,
            credential=AzureKeyCredential(DI_KEY)
        )

        # Read PDF content
        pdf_data = inputBlob.read()

        # Analyze document with custom model
        poller = document_intelligence_client.begin_analyze_document(
            model_id=MODEL_ID,
            body=pdf_data,
            content_type="application/pdf"
        )
        result = poller.result()
        analyze_result_dict = serialize_analyze_result(result)

        # Save Document Intelligence JSON to form-json container
        json_blob_name = inputBlob.name.replace("form-pdfs/", "").replace(".pdf", ".json")
        save_json_to_blob(blob_service_client, "form-json", json_blob_name, analyze_result_dict)
        jsonOutput.set(json.dumps(analyze_result_dict, indent=2))

        # Extract fields
        fields = result.documents[0].fields

        # Validate form fields
        issues = validate_form_fields(fields, result)

        # Analyze signatures with GPT-4o
        signatures, signature_issues = analyze_signatures_with_gpt4o(fields, result, pdf_data, inputBlob.name)
        issues.extend(signature_issues)

        # Prepare report JSON
        report_data = {
            "form_id": inputBlob.name,
            "status": "success",
            "blob_name": inputBlob.name,
            "signatures": signatures,
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }

        # Save report to form-reports container
        report_blob_name = inputBlob.name.replace("form-pdfs/", "").replace(".pdf", "_report.json")
        save_json_to_blob(blob_service_client, "form-reports", report_blob_name, report_data)
        reportOutput.set(json.dumps(report_data, indent=2))

        logging.info(f"Successfully processed {inputBlob.name}")

    except Exception as e:
        logging.error(f"Error processing {inputBlob.name}: {str(e)}")
        report_data = {
            "form_id": inputBlob.name,
            "status": "error",
            "blob_name": inputBlob.name,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }
        report_blob_name = inputBlob.name.replace("form-pdfs/", "").replace(".pdf", "_report.json")
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        save_json_to_blob(blob_service_client, "form-reports", report_blob_name, report_data)
        reportOutput.set(json.dumps(report_data, indent=2))
        raise