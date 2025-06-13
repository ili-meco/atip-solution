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
from PIL import Image
import io

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

def extract_signature_image(pdf_data: bytes, page_num: int, bounding_box: list, field_name: str, blob_service_client: BlobServiceClient, blob_name: str) -> Optional[Dict]:
    """Extract a signature image from a PDF using PyMuPDF, resize it, and save to form-signatures."""
    try:
        # Open the PDF document
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        page = pdf_document[page_num - 1]  # Page numbers are 1-based from DI, 0-based in PyMuPDF
        page_rect = page.rect  # Page dimensions in points (1 inch = 72 points)
        logging.info(f"Page {page_num} dimensions: {page_rect}")

        # Initialize flag to determine if fallback region should be used
        use_fallback = False

        # Validate bounding box: must be a list of at least 8 floats representing 4 (x, y) coordinates
        if not isinstance(bounding_box, list) or len(bounding_box) < 8 or not all(isinstance(x, (int, float)) for x in bounding_box):
            logging.warning(f"Invalid bounding box for field '{field_name}' on page {page_num}: {bounding_box}. Using fallback region.")
            use_fallback = True
        else:
            # Convert DI bounding box (in inches) to points (1 inch = 72 points)
            coordinates = [(bounding_box[i], bounding_box[i+1]) for i in range(0, len(bounding_box), 2)]
            coordinates_points = [(x * 72, y * 72) for x, y in coordinates]
            x0 = min(p[0] for p in coordinates_points)
            y0 = min(p[1] for p in coordinates_points)
            x1 = max(p[0] for p in coordinates_points)
            y1 = max(p[1] for p in coordinates_points)

            # Validate bounding box
            if x0 >= x1 or y0 >= y1:
                logging.warning(f"Invalid bounding box dimensions for field '{field_name}' on page {page_num}: x0={x0}, x1={x1}, y0={y0}, y1={y1}. Using fallback region.")
                use_fallback = True
            elif x0 < 0 or y0 < 0 or x1 > page_rect.width or y1 > page_rect.height:
                logging.warning(f"Bounding box outside page bounds for field '{field_name}' on page {page_num}: ({x0}, {y0}, {x1}, {y1}). Using fallback region.")
                use_fallback = True
            elif x0 < 36 and y0 < 36:  # Top-left corner (within 0.5 inches = 36 points)
                logging.warning(f"Bounding box in top-left corner for field '{field_name}' on page {page_num}: ({x0}, {y0}). Using fallback region.")
                use_fallback = True

        # Define the extraction rectangle
        if use_fallback:
            # Fallback region: bottom 10% of the page (lowered as requested)
            page_width = page_rect.width
            page_height = page_rect.height
            x_start = 36  # 0.5-inch margin from left (36 points)
            x_end = page_width - 36  # 0.5-inch margin from right
            y_start = page_height * 0.90  # Start at 90% down the page
            y_end = page_height - 36  # 0.5-inch margin from bottom
            rect = fitz.Rect(x_start, y_start, x_end, y_end)
            logging.info(f"Fallback region for field '{field_name}' on page {page_num}: {rect}")
        else:
            # Use the provided bounding box (already converted to points)
            rect = fitz.Rect(x0, y0, x1, y1)
            logging.info(f"Using provided bounding box for field '{field_name}' on page {page_num}: {rect}")

        # Add padding (20 points) to capture the full signature
        pad = 20
        rect = fitz.Rect(
            max(page_rect.x0, rect.x0 - pad),
            max(page_rect.y0, rect.y0 - pad),
            min(page_rect.x1, rect.x1 + pad),
            min(page_rect.y1, rect.y1 + pad)
        )
        logging.info(f"Padded rectangle for field '{field_name}' on page {page_num}: {rect}")

        # Render the page at 150 DPI
        dpi = 150
        matrix = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for rendering
        pix = page.get_pixmap(matrix=matrix, clip=rect)
        img_bytes = pix.tobytes("png")
        logging.info(f"Rendered pixmap for field '{field_name}' on page {page_num}: width={pix.width}, height={pix.height}")

        # Resize the image to a maximum dimension of 512 pixels while preserving aspect ratio
        img = Image.open(io.BytesIO(img_bytes))
        max_size = 512
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, format="PNG")
        img_bytes_resized = output.getvalue()

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes_resized).decode("utf-8")
        base64_length = len(img_base64)
        estimated_tokens = base64_length // 4  # Rough estimate: ~1 token per 4 characters
        logging.info(f"Image for field '{field_name}' on page {page_num}: base64 length={base64_length}, estimated tokens={estimated_tokens}")

        # Determine the section for the filename
        if "2. Applicant's Information" in field_name:
            section = "applicant"
        elif "Related Individual's Information" in field_name:
            section = field_name.split(" ")[0].replace(".", "_") + "_related_individual"
        else:
            section = "unknown"

        # Save the PNG to form-signatures
        signature_blob_name = f"{blob_name.replace('form-pdfs/', '').replace('.pdf', '')}_{section}_page{page_num}.png"
        blob_client = blob_service_client.get_blob_client(container="form-signatures", blob=signature_blob_name)
        try:
            logging.info(f"Uploading signature image to form-signatures/{signature_blob_name} ...")
            blob_client.upload_blob(img_bytes_resized, overwrite=True)
            logging.info(f"Successfully uploaded signature image to form-signatures/{signature_blob_name}")
        except Exception as upload_exc:
            logging.error(f"Failed to upload signature image to form-signatures/{signature_blob_name}: {str(upload_exc)}")
    except Exception as e:
        logging.error(f"Failed to extract signature image for field '{field_name}' on page {page_num}: {str(e)}")
        if 'pdf_document' in locals():
            pdf_document.close()
        return None

def detect_ink_color_with_gpt4o(client: AzureOpenAI, signature_images: List[Dict]) -> List[Dict]:
    """Use GPT-4o to detect the ink color of signatures in the provided images."""
    signatures_with_colors = []
    
    for sig in signature_images:
        section = sig["section"]
        page = sig["page"]
        img_base64 = sig["image"]

        # Map the section to the correct form section name for the prompt
        if section == "applicant":
            section_display = "2. Applicant's Information"
        elif "related_individual" in section:
            section_num = section.split("_")[0]
            section_display = f"{section_num} Related Individual’s Information"
        else:
            section_display = section

        # Prompt explicitly mentioning "json format"
        prompt = f"""
You are analyzing a PNG image of a handwritten signature from an IMM 5744 E consent form.

The signature appears in section {section_display} on page {page}.

Your task is to determine the *visible ink color* used in the signature. Focus on actual color hues in the image—do not guess or infer contextually.

You must choose only one of the following options:
- "blue"
- "black"
- "red"
- "other"
You must label each image into one of these categories. If you are not sure, pick the closest color. If it's in between blue and other color, pick blue

Return your answer strictly in this JSON format:
{{"color_detected": "<color>"}}

**Image (base64-encoded PNG):**
![Signature](data:image/png;base64,{img_base64})
"""

        # Estimate token count (rough approximation)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # ~1 token per 4 characters
        logging.info(f"Prompt for section {section_display}, page {page}: length={prompt_length}, estimated tokens={estimated_tokens}")

        if estimated_tokens > 128000:
            logging.error(f"Prompt for section {section_display}, page {page} exceeds token limit: {estimated_tokens} tokens")
            color_detected = "other"
        else:
            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=100
                )
                result = json.loads(response.choices[0].message.content)
                color_detected = result.get("color_detected", "other")
                logging.info(f"GPT-4o ink color detection for section {section_display}, page {page}: {color_detected}")
            except Exception as e:
                logging.error(f"Error detecting ink color for section {section_display}, page {page}: {str(e)}")
                color_detected = "other"

        signatures_with_colors.append({
            "section": section,
            "page": page,
            "color_detected": color_detected,
            "is_blue": color_detected == "blue"
        })

    return signatures_with_colors

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

def analyze_signatures_with_gpt4o(fields: Dict, doc_result: AnalyzeResult, pdf_data: bytes, blob_name: str, blob_service_client: BlobServiceClient) -> tuple:
    """Use Azure OpenAI GPT-4o to analyze signatures for compliance, with ink color detection in a separate call."""
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
    signature_field_mapping = {}  # To map sections to field names for issue reporting
    
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
                signature_value = fields[field].get("valueSignature")
                is_present = signature_value == "signed"
                bounding_region = fields[field].get("boundingRegions", [{}])[0]
                page = bounding_region.get("pageNumber", 1)
                bounding_box = bounding_region.get("polygon", [])
                logging.info(f"[Applicant] Signature field '{field}': valueSignature={signature_value}, page={page}, bounding_box={bounding_box}")
                extracted_data["applicant"]["signature"] = {
                    "present": is_present,
                    "page": page,
                    "bounding_box": bounding_box
                }
                signature_field_mapping["applicant"] = field
                # Always attempt extraction, even if not signed
                if isinstance(bounding_box, list) and len(bounding_box) >= 8:
                    logging.info(f"[Applicant] Attempting to extract signature image for field '{field}' on page {page}")
                    result = extract_signature_image(pdf_data, page, bounding_box, field, blob_service_client, blob_name)
                    if result:
                        signature_images.append({
                            "section": result["section"],
                            "image": result["image"],
                            "page": page
                        })
                        logging.info(f"[Applicant] Signature image extracted and appended for field '{field}' on page {page}")
                    else:
                        logging.warning(f"[Applicant] No valid signature image extracted for field '{field}' on page {page}")
                        extracted_data["applicant"]["signature"]["is_blue"] = None
                else:
                    logging.warning(f"[Applicant] Invalid or missing bounding box for field '{field}' on page {page}: {bounding_box}")
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
                    signature_value = fields[field].get("valueSignature")
                    is_present = signature_value == "signed"
                    bounding_region = fields[field].get("boundingRegions", [{}])[0]
                    page = bounding_region.get("pageNumber", 1)
                    bounding_box = bounding_region.get("polygon", [])
                    logging.info(f"[Related {section}] Signature field '{field}': valueSignature={signature_value}, page={page}, bounding_box={bounding_box}")
                    extracted_data["related_individuals"][i]["signature"] = {
                        "present": is_present,
                        "page": page,
                        "bounding_box": bounding_box
                    }
                    section_key = f"related_individual_{i}"
                    signature_field_mapping[section_key] = field
                    # Always attempt extraction, even if not signed
                    if isinstance(bounding_box, list) and len(bounding_box) >= 8:
                        logging.info(f"[Related {section}] Attempting to extract signature image for field '{field}' on page {page}")
                        result = extract_signature_image(pdf_data, page, bounding_box, field, blob_service_client, blob_name)
                        if result:
                            signature_images.append({
                                "section": result["section"],
                                "image": result["image"],
                                "page": page
                            })
                            logging.info(f"[Related {section}] Signature image extracted and appended for field '{field}' on page {page}")
                        else:
                            logging.warning(f"[Related {section}] No valid signature image extracted for field '{field}' on page {page}")
                            extracted_data["related_individuals"][i]["signature"]["is_blue"] = None
                    else:
                        logging.warning(f"[Related {section}] Invalid or missing bounding box for field '{field}' on page {page}: {bounding_box}")
                        extracted_data["related_individuals"][i]["signature"]["is_blue"] = None
                else:
                    extracted_data["related_individuals"][i][key] = {"value": fields[field].get("valueString", "")}
            else:
                logging.info(f"[Related {section}] Field '{field}' not found in extracted fields")

    # Step 1: Detect ink color with a separate GPT-4o call
    signatures = detect_ink_color_with_gpt4o(client, signature_images)

    # Step 2: Update extracted_data with ink color results
    for sig in signatures:
        section = sig["section"]
        is_blue = sig["is_blue"]
        if section == "applicant":
            if "signature" in extracted_data["applicant"]:
                extracted_data["applicant"]["signature"]["is_blue"] = is_blue
            else:
                logging.warning(f"Signature data missing for applicant section; cannot set is_blue")
        elif "related_individual" in section:
            try:
                # Extract the section number (e.g., "2_1_related_individual" -> "2_1")
                section_num = section.split("_related_individual")[0]  # "2_1"
                minor_num = int(section_num.split("_")[1])  # "1" from "2_1"
                idx = minor_num - 1  # Map 1->0, 2->1, 3->2
                logging.info(f"Mapping section '{section}' to related_individuals index {idx}")
                # Ensure idx is within bounds
                if 0 <= idx < len(extracted_data["related_individuals"]):
                    if "signature" not in extracted_data["related_individuals"][idx]:
                        logging.warning(f"Signature data missing for related_individuals[{idx}]; initializing with is_blue")
                        extracted_data["related_individuals"][idx]["signature"] = {"is_blue": is_blue}
                    else:
                        extracted_data["related_individuals"][idx]["signature"]["is_blue"] = is_blue
                else:
                    logging.error(f"Index {idx} out of bounds for related_individuals (length {len(extracted_data['related_individuals'])})")
            except (IndexError, ValueError) as e:
                logging.error(f"Error processing section '{section}' for ink color update: {str(e)}")
        else:
            logging.warning(f"Unknown section '{section}' in ink color detection results")

    # Step 3: Compliance check with GPT-4o (without ink color detection)
    prompt = f"""
    You are an expert in analyzing the IMM 5744 E consent form for ATIP compliance. Perform these tasks based solely on the provided input data, without making assumptions or inferring missing information:
    **ATIP Consent Form Instructions**:
    - The form has three main sections:
      - Section 1: Designated Representative’s Information (requester details).
      - Section 2: Applicant’s Information.
      - Sections 2.1 to 2.3: Related Individual’s Information (up to 3 related individuals).
    - Minors under 16 require both parents’ signatures, or a court order.
    - Individuals 16 and older must have their own signature.
    - Signatures must be original, include the date, and cannot be electronic.
    - Required fields (family_name, given_name, dob) must be present and non-empty for any section with data (e.g., if any field in a section is filled, all required fields must be filled).
    **Tasks**:
    1. **Compliance Check**:
       - Verify these rules:
         - Individuals 16+ must have their own signature.
         - Minors (<16) must have signatures from both parents or a court order.
         - Signatures must include a date (check form_metadata.submission_date if available).
         - Required fields (family_name, given_name, dob) must be present and non-empty for any section with data.
       - For each issue, specify the page number, section name (e.g., '2. Applicant's Information'), and individual (e.g., 'Applicant Jason Jenna (age 1)').
       - Provide clear action recommendations (e.g., 'Request missing signature', 'Verify court order for minor').
       - Do not report issues about exceeding the limit of related individuals; focus only on field inputs and signatures.
       - Do NOT report issues related to ink color; ink color has already been checked, and the provided 'is_blue' value in the input data should be used for reference only.
    **Input Data**:
    {json.dumps(extracted_data, indent=2)}
    **Output a JSON object with**:
    - 'form_id': '{blob_name}'
    - 'issues': List of objects with 'description' (specific, e.g., 'Minor under 16 requires both parents’ signatures in 2. Applicant's Information'), 'field' (e.g., 'Signature'), 'page' (integer), and 'action' (e.g., 'Request both parents’ signatures'). Exclude any issues about exceeding related individuals limit or ink color.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        result = json.loads(response.choices[0].message.content)
        logging.info(f"GPT-4o compliance check response: {json.dumps(result, indent=2)}")
        
        # Filter out issues related to exceeding related individuals limit
        compliance_issues = [
            issue for issue in result.get("issues", [])
            if "exceeded limit of related individuals" not in issue.get("description", "").lower()
        ]
    except Exception as e:
        logging.error(f"Error analyzing signatures with GPT-4o for compliance: {str(e)}")
        compliance_issues = [{"description": f"Failed to analyze signatures: {str(e)}", "field": "Signature Analysis", "page": 1, "action": "Flag for review"}]

    # Step 4: Add ink color issues for signatures that are present but not blue
    ink_issues = []
    for sig in signatures:
        section = sig["section"]
        is_blue = sig["is_blue"]
        page = sig["page"]
        field = signature_field_mapping.get(section, "Unknown Field")
        section_name = "2. Applicant's Information" if section == "applicant" else f"{section.replace('_related_individual', '').replace('_', '.')} Related Individual’s Information"
        
        # Only add an issue if the signature is present (image exists) and not blue
        if sig["color_detected"] != "other" and not is_blue:
            ink_issues.append({
                "description": f"Signature not in blue ink in {section_name}",
                "field": field,
                "page": page,
                "action": "Request signature in blue ink"
            })

    # Combine signatures for output (remove color_detected to match original format)
    signatures_output = [
        {
            "section": sig["section"],
            "is_blue": sig["is_blue"],
            "page": sig["page"]
        }
        for sig in signatures
    ]

    # Combine all issues
    all_issues = compliance_issues + ink_issues
    return signatures_output, all_issues

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
        signatures, signature_issues = analyze_signatures_with_gpt4o(fields, result, pdf_data, inputBlob.name, blob_service_client)
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