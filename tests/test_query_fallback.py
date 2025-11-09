import os
import importlib
from pathlib import Path

PDF_CONTENT = "Sample certificate: This document certifies completion of the course." * 3


def write_pdf(tmp_dir: Path):
    from fpdf import FPDF
    pdf_path = tmp_dir / 'certificate.pdf'
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, PDF_CONTENT)
    pdf.output(str(pdf_path))
    return pdf_path


def prepare_app(tmp_dir: Path):
    os.environ['API_KEY'] = ''
    os.environ['DATA_DIR'] = str(tmp_dir)
    os.environ['LLM_PROVIDER'] = 'huggingface'
    # Use a trivial HF model name placeholder (actual call will fail without token, so mock later)
    os.environ['HF_MODEL'] = 'google/flan-t5-base'
    # Mock embeddings: patch embed_texts to return random small vectors
    import numpy as np
    def fake_embed(texts):
        return np.random.rand(len(texts), 16).astype('float32')
    if 'backend.services.embeddings' in globals():
        import backend.services.embeddings as emb_mod
        emb_mod.embed_texts = fake_embed
    else:
        import backend.services.embeddings as emb_mod
        emb_mod.embed_texts = fake_embed
    if 'backend.app' in globals():
        import backend.app as app_mod
        import os
        import importlib
        from pathlib import Path

        PDF_CONTENT = "Sample certificate: This document certifies completion of the course." * 3

        def write_pdf(tmp_dir: Path):
            # Minimal PDF writer without external dependency: write a single-page PDF structure.
            pdf_path = tmp_dir / 'certificate.pdf'
            content = PDF_CONTENT.replace('\n', ' ')
            # Very basic PDF (not robust) sufficient for pypdf to parse text if needed.
            raw = f"%PDF-1.1\n1 0 obj<<>>endobj\n2 0 obj<<>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>endobj\n4 0 obj<< /Length {len(content)+33} >>stream\nBT /F1 12 Tf 50 750 Td ({content}) Tj ET\nendstream\nendobj\n5 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n6 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n7 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 612 792] >>endobj\nxref\n0 8\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000110 00000 n \n0000000200 00000 n \n0000000300 00000 n \n0000000350 00000 n \n0000000410 00000 n \ntrailer<< /Size 8 /Root 5 0 R >>\nstartxref\n480\n%%EOF"
            with open(pdf_path, 'wb') as f:
                f.write(raw.encode('latin-1', errors='ignore'))
            return pdf_path
    assert 'answer' in data
    assert isinstance(data.get('retrieved'), list)
    assert len(data['retrieved']) <= 2
