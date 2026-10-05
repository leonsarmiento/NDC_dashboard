"""Strict output schema for LLM returns, plus a normaliser.

Every document is processed independently and the LLM is forced to emit ONE
JSON object per document with the shape below. The normaliser guarantees that
even if a model omits a field, the saved object always has the exact same keys,
so the sidekick can merge every session's output into one tidy table.

Shape
-----
{
  "document": {
    "country": str,
    "document_title_registry": str,
    "document_title_printed": str,
    "document_version": str,
    "document_status": str,
    "submission_date": str,
    "original_language": str
  },
  "passages": [
    {
      "page": str,
      "location_in_document": str,
      "line_number_on_page": str,
      "original_text": str,
      "english_translation": str,
      "commitment_type": "Restoration" | "Conservation" | "Agriculture" | "Both",
      "restoration_concepts": str,
      "relevance_confidence": "High" | "Medium" | "Low",
      "target_value": str,
      "target_unit": str,
      "notes": str
    },
    ...
  ]
}

Design notes
------------
* All values are stored as strings (including "Target Value" / "Target Unit")
  so that mixed content (e.g. "NA", "20%", "1,000,000") never breaks the table.
* ``"NA"`` is the canonical "not available / not applicable" sentinel.
* Each passage is the atomic row of the final database (long / tidy format).
"""

from __future__ import annotations

NA = "NA"

DOCUMENT_FIELDS = [
    "country",
    "document_title_registry",
    "document_title_printed",
    "document_version",
    "document_status",
    "submission_date",
    "original_language",
]

PASSAGE_FIELDS = [
    "page",
    "location_in_document",
    "line_number_on_page",
    "original_text",
    "english_translation",
    "commitment_type",
    "restoration_concepts",
    "relevance_confidence",
    "target_value",
    "target_unit",
    "notes",
]

# Column order used by the sidekick when flattening to Excel.
# (document fields first, then passage fields, then a source-file trace column)
EXCEL_COLUMNS = (
    ["source_file", "provider", "model"]
    + [f"doc_{f}" for f in DOCUMENT_FIELDS]
    + [f"passage_{f}" for f in PASSAGE_FIELDS]
)

# Human-friendly header labels for the Excel output.
EXCEL_HEADERS = {
    "source_file": "Source file",
    "provider": "Provider",
    "model": "Model",
    "doc_country": "Country",
    "doc_document_title_registry": "Document title (NDC Registry)",
    "doc_document_title_printed": "Document title (as printed on document)",
    "doc_document_version": "Document version",
    "doc_document_status": "Document status",
    "doc_submission_date": "Submission date",
    "doc_original_language": "Original language",
    "passage_page": "Page (printed page no.)",
    "passage_location_in_document": "Location in document (section / clause / table)",
    "passage_line_number_on_page": "Line number on page",
    "passage_original_text": "Original text (verbatim, original language)",
    "passage_english_translation": "English translation",
    "passage_commitment_type": "Commitment type",
    "passage_restoration_concepts": "Restoration-related concept(s)",
    "passage_relevance_confidence": "Relevance confidence",
    "passage_target_value": "Target Value",
    "passage_target_unit": "Target Unit",
    "passage_notes": "Notes / caveats for the auditor",
}


def _clean(value) -> str:
    """Coerce any value to a clean string, mapping None/empty to ``NA``."""
    if value is None:
        return NA
    if isinstance(value, str):
        v = value.strip()
        return v if v else NA
    if isinstance(value, (int, float)):
        # avoid "20.0" style floats; keep ints clean
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def normalize_record(raw) -> dict:
    """Return a record that is guaranteed to match the schema exactly.

    Accepts whatever the model returned (dict, or a dict nested under a key
    such as ``"result"``) and fills every missing field with ``NA``.
    """
    if not isinstance(raw, dict):
        raw = {}

    # Some models wrap the object. Look for the document/passages payload.
    if "document" not in raw and "passages" not in raw:
        for key in ("result", "data", "output", "response", "extraction"):
            if isinstance(raw.get(key), dict) and (
                "document" in raw[key] or "passages" in raw[key]
            ):
                raw = raw[key]
                break

    doc_raw = raw.get("document") if isinstance(raw.get("document"), dict) else {}
    document = {f: _clean(doc_raw.get(f)) for f in DOCUMENT_FIELDS}

    passages_raw = raw.get("passages")
    if not isinstance(passages_raw, list):
        # tolerate a single dict, or an object nested under another key
        if isinstance(passages_raw, dict):
            passages_raw = [passages_raw]
        else:
            for key in ("relevant_passages", "passage", "items", "rows"):
                if isinstance(raw.get(key), list):
                    passages_raw = raw[key]
                    break
            else:
                passages_raw = []

    passages = []
    for p in passages_raw:
        if not isinstance(p, dict):
            continue
        passages.append({f: _clean(p.get(f)) for f in PASSAGE_FIELDS})

    return {"document": document, "passages": passages}


def validate_record(record: dict) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    problems = []
    if not isinstance(record, dict):
        return ["record is not a dict"]
    doc = record.get("document")
    if not isinstance(doc, dict):
        problems.append("missing 'document' object")
    else:
        for f in DOCUMENT_FIELDS:
            if f not in doc:
                problems.append(f"document missing field: {f}")
    passages = record.get("passages")
    if not isinstance(passages, list):
        problems.append("'passages' is not a list")
    else:
        for i, p in enumerate(passages):
            if not isinstance(p, dict):
                problems.append(f"passage[{i}] is not an object")
                continue
            for f in PASSAGE_FIELDS:
                if f not in p:
                    problems.append(f"passage[{i}] missing field: {f}")
    return problems
