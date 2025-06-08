# Document Parsing (`docparser`) Module

This module is responsible for parsing various document formats, extracting their content and structure, and preparing them for further processing within the ApeRAG system, primarily by chunking the content into manageable pieces.

## Core Functionality

The `docparser` module takes a file path as input and returns a list of structured `Part` objects. These parts represent different semantic elements of the document, such as titles, paragraphs, code blocks, images, and tables. It also often generates a full Markdown representation of the document.

## Key Components

1.  **`DocParser` (`doc_parser.py`)**:
    *   Acts as the main entry point and orchestrator for the parsing process.
    *   Manages a configurable list of specialized parsers (e.g., for audio, images, Markdown, complex documents).
    *   Selects the appropriate parser based on file extension and configuration.
    *   Allows enabling/disabling specific parsers and overriding their settings.

2.  **`BaseParser` and `Part` Objects (`base.py`)**:
    *   `BaseParser`: An abstract base class that all individual parsers must implement. It defines a common interface with methods like `supported_extensions()` and `parse_file()`.
    *   `Part`: A Pydantic model representing a segment of the parsed document. There are various specialized `Part` types:
        *   `TextPart`: Plain text content.
        *   `TitlePart`: Document headings, including their level.
        *   `CodePart`: Code blocks, with optional language information.
        *   `MarkdownPart`: Represents the full Markdown content of a document.
        *   `ImagePart`: Represents an image, including its URL (which might be an `asset://` URL) and alternative text.
        *   `MediaPart`: A general part for media.
        *   `AssetBinPart`: Stores binary data for embedded assets (like images extracted from data URIs or complex documents), along with a unique `asset_id` and `mime_type`. These are typically referenced by other parts via `asset://<asset_id>` URLs.
    *   `FallbackError`: An exception raised by parsers if they cannot process a file and want to allow another parser to attempt it.

3.  **Individual Parsers**:
    *   **`MarkItDownParser` (`markitdown_parser.py`)**:
        *   Handles a wide array of formats including `.txt`, `.md`, `.html`, `.ipynb`, `.pdf`, `.epub`, and Microsoft Office documents (`.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt`).
        *   Uses the `markitdown` library for the primary conversion to Markdown.
        *   For older Office formats (`.doc`, `.ppt`), it can use `soffice` (LibreOffice/OpenOffice) to convert them to modern XML formats first.
        *   The resulting Markdown is then processed by `parse_md.py`.
    *   **`DocRayParser` (`docray_parser.py`)**:
        *   Designed for complex, layout-intensive documents like `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`.
        *   Relies on an external "DocRay" microservice (configured via `settings.DOCRAY_HOST`).
        *   Submits documents to DocRay, polls for completion, and retrieves a structured JSON output ("middle_json") along with extracted images.
        *   This `middle_json` provides detailed layout information (pages, blocks, bounding boxes) which is then converted into `Part` objects, including `AssetBinPart` for images.
        *   Also generates a consolidated Markdown representation from these parts.
    *   **`AudioParser` (`audio_parser.py`)**:
        *   Transcribes audio files (e.g., `.mp3`, `.wav`, `.ogg`) to text.
        *   Uses an external Whisper ASR webservice (configured via `settings.WHISPER_HOST`).
        *   Outputs a `TextPart` with the transcription.
    *   **`ImageParser` (`image_parser.py`)**:
        *   Extracts text from images (OCR) for formats like `.jpg`, `.png`, `.bmp`.
        *   Uses an external PaddleOCR service (configured via `settings.PADDLEOCR_HOST`).
        *   Outputs a `TextPart` with the extracted text.

4.  **Markdown Processing (`parse_md.py`)**:
    *   Takes a Markdown string (often the output of `MarkItDownParser` or `DocRayParser`) and converts it into a detailed list of `Part` objects.
    *   Uses `markdown-it-py` for tokenizing the Markdown.
    *   The `PartConverter` class iterates through these tokens, creating corresponding `Part` objects (e.g., `TitlePart`, `TextPart`, `CodePart`, `ImagePart`).
    *   Handles extraction of embedded base64 data URIs: converts them to `AssetBinPart` objects and replaces the URI with an `asset://` link in the Markdown.
    *   Preserves source mapping (line numbers) from the Markdown to the `Part` objects.

5.  **Utility Functions (`utils.py`)**:
    *   `convert_office_doc()`: Uses `soffice` to convert between Office document formats.
    *   `get_soffice_cmd()`: Checks for the `soffice` command.
    *   `asset_bin_part_to_url()`: Creates the `asset://` URL for an `AssetBinPart`.
    *   `extension_to_mime_type()`: Maps file extensions to MIME types.

6.  **Chunking (`chunking.py`)**:
    *   This crucial component takes the list of `Part` objects produced by a parser and further processes them into chunks of a specified token size, suitable for embedding and retrieval in a RAG pipeline.
    *   **`Rechunker`**:
        *   Groups parts based on document titles to maintain semantic context.
        *   Merges and splits parts to adhere to `chunk_size` and `chunk_overlap` constraints, using a provided `tokenizer`.
        *   Preserves and propagates title hierarchies and source map information (from Markdown or PDF) into the final chunks.
    *   **`SimpleSemanticSplitter`**:
        *   Used by `Rechunker` to break down individual text segments that are too large.
        *   Employs a hierarchical list of separators (paragraph breaks, line breaks, sentence terminators, etc.) to split text while attempting to preserve meaning.
        *   Implements overlap between chunks.

## Workflow

1.  A file path is provided to `DocParser`.
2.  `DocParser` selects an appropriate parser (e.g., `MarkItDownParser`, `DocRayParser`).
3.  The selected parser processes the file:
    *   It might convert the file to Markdown (e.g., `MarkItDownParser`).
    *   It might call external services for complex parsing or OCR/ASR (`DocRayParser`, `ImageParser`, `AudioParser`).
4.  The (potentially intermediate Markdown) content is parsed into a list of `Part` objects using `parse_md.py` if applicable. This step structures the document into semantic units like paragraphs, titles, code blocks, and handles embedded images.
5.  The resulting list of `Part` objects is then passed to the `rechunk` function in `chunking.py`.
6.  `rechunk` intelligently combines and splits these parts into final text chunks of a desired token size, ready for embedding. Metadata, including title hierarchy and source location, is associated with each chunk.

This module forms the foundation of the ingestion pipeline in ApeRAG, ensuring that diverse document types can be effectively processed and structured for retrieval and generation tasks. 