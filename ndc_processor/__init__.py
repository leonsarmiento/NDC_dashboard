"""NDC Document Processor.

A small pipeline that:
  1. Converts curated NDC documents to text with MarkItDown.
  2. Sends each document (with an instructions prompt) to an LLM API.
  3. Forces the LLM to return a strict JSON object.
  4. Saves one JSON file per document.
  5. (Sidekick) Merges all JSON returns into a single tabular database (Excel).

The app is designed to run on Windows machines inside a dedicated
conda environment named ``doc_processor``. All file handling uses
``pathlib.Path`` and cross-platform-safe filename sanitisation.
"""

__version__ = "0.1.0"
