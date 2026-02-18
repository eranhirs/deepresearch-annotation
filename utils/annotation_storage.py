"""Google Sheets backend for storing human annotations.

Requires:
- st.secrets["gcp_service_account"] with service account credentials
- st.secrets["annotations_sheet_name"] with the Google Sheet name

Sheet schema (one row per annotated segment):
annotator_id | model_name | example_id | segment_idx | sentence_text |
is_repetitive | repetitive_explanation | is_non_coherent | non_coherent_explanation |
is_over_specific | over_specific_explanation | is_missing_details |
missing_details_explanation | other_issues | timestamp
"""

import json
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st


SHEET_HEADERS = [
    "annotator_id", "model_name", "example_id", "segment_idx", "sentence_text",
    "is_repetitive", "repetitive_explanation",
    "is_non_coherent", "non_coherent_explanation",
    "is_over_specific", "over_specific_explanation",
    "is_missing_details", "missing_details_explanation",
    "other_issues", "timestamp",
]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource
def get_gsheet_client() -> gspread.Client:
    """Authenticate and return a gspread client using service account from st.secrets."""
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(dict(creds_info), scopes=SCOPES)
    return gspread.authorize(creds)


def _get_worksheet() -> gspread.Worksheet:
    """Get the annotations worksheet, creating headers if the sheet is empty."""
    client = get_gsheet_client()
    sheet_name = st.secrets["annotations_sheet_name"]
    spreadsheet = client.open(sheet_name)
    worksheet = spreadsheet.sheet1

    # Initialize headers if sheet is empty
    existing = worksheet.row_values(1)
    if not existing:
        worksheet.update("A1", [SHEET_HEADERS])

    return worksheet


def _rows_to_dicts(all_values: list[list[str]]) -> list[dict[str, str]]:
    """Convert get_all_values() output to list of dicts using first row as headers."""
    if len(all_values) < 2:
        return []
    headers = all_values[0]
    results = []
    for row in all_values[1:]:
        # Pad short rows with empty strings
        padded = row + [""] * max(0, len(headers) - len(row))
        results.append(dict(zip(headers, padded)))
    return results


def load_annotations(annotator_id: str, model_name: str, example_id: str) -> dict[str, dict]:
    """Load existing annotations for this annotator + model + example.

    Returns {segment_idx: annotation_dict}.
    """
    worksheet = _get_worksheet()
    all_values = worksheet.get_all_values()
    all_records = _rows_to_dicts(all_values)

    annotations = {}
    for row in all_records:
        if (row.get("annotator_id") == annotator_id
                and row.get("model_name") == model_name
                and row.get("example_id") == str(example_id)):
            seg_idx = str(row.get("segment_idx", ""))
            annotations[seg_idx] = {
                "is_repetitive": row.get("is_repetitive", ""),
                "repetitive_explanation": row.get("repetitive_explanation", ""),
                "is_non_coherent": row.get("is_non_coherent", ""),
                "non_coherent_explanation": row.get("non_coherent_explanation", ""),
                "is_over_specific": row.get("is_over_specific", ""),
                "over_specific_explanation": row.get("over_specific_explanation", ""),
                "is_missing_details": row.get("is_missing_details", ""),
                "missing_details_explanation": row.get("missing_details_explanation", ""),
                "other_issues": row.get("other_issues", ""),
                "timestamp": row.get("timestamp", ""),
            }
    return annotations


def save_annotation(
    annotator_id: str,
    model_name: str,
    example_id: str,
    segment_idx: str,
    sentence_text: str,
    annotation: dict,
) -> None:
    """Save or update an annotation row in the sheet.

    Composite key: (annotator_id, model_name, example_id, segment_idx).
    If a row with this key exists, it is updated; otherwise a new row is appended.
    """
    worksheet = _get_worksheet()
    timestamp = datetime.now(timezone.utc).isoformat()

    row_data = [
        annotator_id,
        model_name,
        str(example_id),
        str(segment_idx),
        sentence_text[:500],  # truncate to avoid sheet cell limits
        str(annotation.get("is_repetitive", "")),
        annotation.get("repetitive_explanation", ""),
        str(annotation.get("is_non_coherent", "")),
        annotation.get("non_coherent_explanation", ""),
        str(annotation.get("is_over_specific", "")),
        annotation.get("over_specific_explanation", ""),
        str(annotation.get("is_missing_details", "")),
        annotation.get("missing_details_explanation", ""),
        annotation.get("other_issues", ""),
        timestamp,
    ]

    # Find existing row with same composite key
    all_values = worksheet.get_all_values()
    target_row = None
    for i, row in enumerate(all_values[1:], start=2):  # skip header
        if (len(row) >= 4
                and row[0] == annotator_id
                and row[1] == model_name
                and row[2] == str(example_id)
                and row[3] == str(segment_idx)):
            target_row = i
            break

    if target_row:
        cell_range = f"A{target_row}:{chr(64 + len(SHEET_HEADERS))}{target_row}"
        worksheet.update(cell_range, [row_data])
    else:
        worksheet.append_row(row_data, value_input_option="RAW")


def download_annotations(annotator_id: str) -> list[dict]:
    """Fetch all annotations for a given annotator as a list of dicts."""
    worksheet = _get_worksheet()
    all_values = worksheet.get_all_values()
    all_records = _rows_to_dicts(all_values)
    return [row for row in all_records if row.get("annotator_id") == annotator_id]
