from pathlib import Path

import pikepdf
import pytest

from aperag.docparser.base import ParserChainError
from aperag.docparser.doc_parser import DocParser
from aperag.domains.indexing.document_parser import document_parser


def _write_minimal_pdf(path: Path, text: str = "PDF Diagnostic Smoke") -> None:
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream\nendobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    startxref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(pdf)


@pytest.mark.parametrize(
    ("filename", "payload", "detail_snippet"),
    [
        ("broken.docx", b"not a zip container", "valid OOXML package"),
        ("broken.xlsx", b"not a zip container", "valid OOXML package"),
        ("broken.pptx", b"not a zip container", "valid OOXML package"),
        ("broken.pdf", b"not a real pdf", "%PDF file header"),
    ],
)
def test_doc_parser_reports_corrupted_binary_documents_with_fixed_diagnostics(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    detail_snippet: str,
) -> None:
    sample = tmp_path / filename
    sample.write_bytes(payload)

    parser = DocParser(parser_config={"use_markitdown": True, "use_mineru": False})

    with pytest.raises(ParserChainError) as exc_info:
        parser.parse_file(sample, metadata={"source": filename})

    assert exc_info.value.source == "runtime"
    assert exc_info.value.code == "parser_chain_failed"
    assert exc_info.value.attempts[0].parser_name == "markitdown"
    assert exc_info.value.attempts[0].code == "corrupted_document"
    assert detail_snippet in (exc_info.value.attempts[0].detail or "")


def test_doc_parser_reports_encrypted_pdf_as_diagnostic_failure(tmp_path: Path) -> None:
    plain_pdf = tmp_path / "plain.pdf"
    encrypted_pdf = tmp_path / "encrypted.pdf"
    _write_minimal_pdf(plain_pdf, text="Secret PDF")

    with pikepdf.open(plain_pdf) as pdf:
        pdf.save(encrypted_pdf, encryption=pikepdf.Encryption(owner="owner-secret", user="user-secret", R=4))

    parser = DocParser(parser_config={"use_markitdown": True, "use_mineru": False})

    with pytest.raises(ParserChainError) as exc_info:
        parser.parse_file(encrypted_pdf, metadata={"source": "encrypted.pdf"})

    assert exc_info.value.code == "parser_chain_failed"
    assert exc_info.value.attempts[0].parser_name == "markitdown"
    assert (
        "password" in (exc_info.value.detail or "").lower()
        or "pdfpasswordincorrect" in (exc_info.value.detail or "").lower()
    )


def test_document_parser_keeps_empty_text_documents_as_empty_content(tmp_path: Path) -> None:
    sample = tmp_path / "empty.txt"
    sample.write_text("", encoding="utf-8")

    result = document_parser.process_document_parsing(str(sample), {"source": "empty.txt"})

    assert result.content == ""
    assert result.doc_parts == []
