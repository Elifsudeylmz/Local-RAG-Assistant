import re

import pymupdf


def _open_pdf(pdf_file):
    """
    pdf_file bir dosya yolu (str) ya da bir Streamlit UploadedFile
    nesnesi olabilir. Her iki durumu da destekler.
    """

    if isinstance(pdf_file, str):
        return pymupdf.open(pdf_file)

    if hasattr(pdf_file, "getvalue"):
        data = pdf_file.getvalue()
    else:
        data = pdf_file.read()

    return pymupdf.open(stream=data, filetype="pdf")


def _clean_text(text):
    """
    PDF'den gelen metindeki satır sonu kırpmalarını ve fazla
    boşlukları temizler. Böylece chunk'lar cümle bütünlüğünü korur
    ve embedding kalitesi artar.
    """

    # Satır sonunda tireyle bölünmüş kelimeleri birleştir
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Tüm satır sonlarını ve tekrar eden boşlukları tek boşluğa indir
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_file):
    """
    PDF dosyasındaki metni sayfa sayfa çıkarır.
    """

    doc = _open_pdf(pdf_file)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = _clean_text(page.get_text())

        if text:
            pages.append({"page": page_number, "text": text})

    doc.close()

    return pages


def create_chunks(pages, chunk_size=1000, overlap=150, min_chunk_len=60):
    """
    Metni chunk'lara böler. Her chunk hangi sayfadan geldiğini korur.

    Kesim noktası öncelikle cümle sonunda (. ? !), bulunamazsa
    kelime sınırında seçilir. Çok kısa chunk'lar (sayfa numarası,
    başlık artığı vb.) atılır.
    """

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]
        length = len(text)

        start = 0

        while start < length:

            end = min(start + chunk_size, length)

            if end < length:

                # Chunk'ın son %40'lık kısmında cümle sonu ara
                window_start = start + int(chunk_size * 0.6)

                cut = max(
                    text.rfind(". ", window_start, end),
                    text.rfind("? ", window_start, end),
                    text.rfind("! ", window_start, end),
                )

                if cut != -1:
                    end = cut + 1
                else:
                    space = text.rfind(" ", start, end)

                    if space > start:
                        end = space

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= min_chunk_len:
                chunks.append({"page": page_number, "text": chunk_text})

            if end >= length:
                break

            next_start = end - overlap

            if next_start <= start:
                next_start = end

            # Overlap başlangıcını kelime başına hizala
            space = text.find(" ", next_start, end)

            if space != -1:
                next_start = space + 1

            start = next_start

    return chunks


if __name__ == "__main__":

    pdf_path = "documents/test.pdf"

    pages = extract_text_from_pdf(pdf_path)
    print(f"\nPDF'deki sayfa sayısı: {len(pages)}")

    chunks = create_chunks(pages)
    print(f"Toplam chunk sayısı: {len(chunks)}")

    if chunks:
        print("\nİlk chunk:")
        print(chunks[0]["text"])
        print(f"\nSayfa: {chunks[0]['page']}")