import unittest
from io import BytesIO

import pymupdf
from docx import Document

from services.document_processor import DocumentProcessingError, process_document, split_into_chunks
from utils import calculate_quiz_score


class DocumentProcessorTests(unittest.TestCase):
    def test_text_upload_extracts_utf8(self):
        result = process_document("notes.txt", "Cell structure".encode("utf-8"))
        self.assertEqual(result.text, "Cell structure")
        self.assertEqual(result.pages, 1)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(DocumentProcessingError):
            process_document("empty.txt", b"  ")

    def test_pdf_upload_extracts_text_and_page_count(self):
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "RAG retrieves relevant context")
        content = pdf.tobytes()
        pdf.close()
        result = process_document("notes.pdf", content)
        self.assertIn("RAG retrieves relevant context", result.text)
        self.assertEqual(result.pages, 1)

    def test_docx_upload_extracts_text(self):
        document = Document()
        document.add_paragraph("Study material")
        stream = BytesIO()
        document.save(stream)
        result = process_document("notes.docx", stream.getvalue())
        self.assertEqual(result.text, "Study material")

    def test_chunking_keeps_overlap(self):
        chunks = split_into_chunks("one two three four five", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five"])

    def test_rejects_unsupported_upload(self):
        with self.assertRaises(DocumentProcessingError):
            process_document("notes.exe", b"data")


class QuizTests(unittest.TestCase):
    def test_score_counts_correct_answers_and_percentage(self):
        questions = [{"answer": "A"}, {"answer": "B"}, {"answer": "C"}]
        correct, percentage = calculate_quiz_score(questions, {0: "A", 1: "wrong"})
        self.assertEqual(correct, 1)
        self.assertAlmostEqual(percentage, 100 / 3)

    def test_empty_quiz_has_zero_score(self):
        self.assertEqual(calculate_quiz_score([], {}), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
