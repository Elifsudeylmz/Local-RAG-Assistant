import hashlib
import json
import os
import re

import numpy as np
import streamlit as st

from pdf_processor import extract_text_from_pdf, create_chunks
from models import get_manager, get_embedding_model, get_chat_model


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# AYARLAR
# ============================================================

# --- CACHE İLE İLGİLİ AYARLAR: DEĞİŞTİRME (değişirse cache geçersiz olur) ---
CACHE_DIR = "embedding_cache"
CACHE_VERSION = "v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# -----------------------------------------------------------------------------

TOP_K = 3

# CPU'da büyük batch'ler zaman aşımına uğrayabilir, küçük tut.
BATCH_SIZE = 16

# Anahtar kelime eşleşmesinin nihai skora katkısı (0 = kapalı).
KEYWORD_WEIGHT = 0.25

# Bir chunk'ın "alakalı" sayılması için gereken minimum EMBEDDING benzerliği (cos).
# Anahtar kelime puanı buna dahil değil; çünkü "güneş/sistem" gibi kelimeler
# alakasız sayfalarda geçince toplam skoru şişiriyordu.
# Ölçülen değerler: alakasız soru (Güneş sistemi) en fazla ~0.39, dokümandaki
# soru (Girit) 0.55+ . Dokümandaki sorular reddediliyorsa düşür (ör. 0.40),
# alakasız sorular cevaplanıyorsa yükselt (ör. 0.50).
MIN_COS = 0.45

NOT_FOUND_TEXT = "Bu yanıt dokümanda bulunmuyor."

# Modele verilen orijinal cümle (prompt'a dokunmamak için değiştirme).
PROMPT_NOT_FOUND = "Bu bilgi dokümanda yer almamaktadır."

# Qwen3-Embedding sorgu tarafında talimat öneki ile daha iyi çalışır.
# Doküman chunk'ları önekle gömülmez.
QUERY_PREFIX = (
    "Instruct: Given a question, retrieve the document passages "
    "that answer the question\nQuery: "
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

for key in ("embedded_data", "file_hash", "processed_file"):
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# MODELLER (bir kez yüklenir, rerun'larda yeniden yüklenmez)
# ============================================================

@st.cache_resource(show_spinner="Embedding modeli hazırlanıyor...")
def load_embedding_resources():

    manager = get_manager()
    model = get_embedding_model(manager)

    if not model.is_loaded:
        model.load()

    return model.get_embedding_client(), str(model.id)


def _configure_chat_client(client):
    """Düşük temperature ve sınırlı çıktı uzunluğu (desteklenmiyorsa atlanır)."""

    for name, value in (
        ("temperature", 0.1),
        ("max_tokens", 400),
    ):
        try:
            setattr(client.settings, name, value)
        except Exception:
            pass


@st.cache_resource(show_spinner="Chat modeli hazırlanıyor...")
def load_chat_client():

    manager = get_manager()
    model = get_chat_model(manager)

    if not model.is_loaded:
        model.load()

    client = model.get_chat_client()

    _configure_chat_client(client)

    return client


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def normalize(matrix):

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim == 1:
        matrix = matrix[None, :]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return matrix / norms


def calculate_file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def get_cache_key(file_hash, model_id):

    raw = f"{file_hash}|{model_id}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{CACHE_VERSION}"

    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_paths(cache_key):

    base = os.path.join(CACHE_DIR, cache_key)

    return base + ".npy", base + ".json"


def load_embedding_cache(cache_key):

    npy_path, json_path = _cache_paths(cache_key)

    if not (os.path.exists(npy_path) and os.path.exists(json_path)):
        return None

    try:
        embeddings = np.load(npy_path)

        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if len(chunks) != len(embeddings):
            return None

        return {"chunks": chunks, "embeddings": embeddings}

    except Exception:
        return None


def save_embedding_cache(cache_key, chunks, embeddings):

    npy_path, json_path = _cache_paths(cache_key)

    np.save(npy_path, embeddings)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


# ============================================================
# EMBEDDING (başarısız batch otomatik ikiye bölünür)
# ============================================================

def _embed_texts(texts, client, max_depth=6):

    if not texts:
        return []

    try:
        response = client.generate_embeddings(texts)
        return [item.embedding for item in response.data]

    except Exception as e:

        if len(texts) == 1 or max_depth <= 0:
            raise RuntimeError(f"Embedding oluşturulamadı: {e}")

        mid = len(texts) // 2

        return (
            _embed_texts(texts[:mid], client, max_depth - 1)
            + _embed_texts(texts[mid:], client, max_depth - 1)
        )


def create_embeddings(chunks, client):

    total = len(chunks)

    progress = st.progress(0)
    status = st.empty()

    vectors = []

    try:
        for start in range(0, total, BATCH_SIZE):

            batch = chunks[start:start + BATCH_SIZE]

            status.text(
                f"Embedding oluşturuluyor... "
                f"{min(start + len(batch), total)}/{total}"
            )

            vectors.extend(_embed_texts([c["text"] for c in batch], client))

            progress.progress(min(start + len(batch), total) / total)

    finally:
        status.empty()
        progress.empty()

    return normalize(vectors)


# ============================================================
# ARAMA (tek matris çarpımı)
# ============================================================

_STOPWORDS = {
    "nedir", "nelerdir", "neler", "nasıl", "hangi", "kaç", "var", "için",
    "nedenleri", "neden", "sırasındaki", "olan", "ile", "kimdir", "nerede",
    "zaman", "hangisi", "olarak", "gibi", "kadar", "daha", "çok",
    "sonuçları", "sonuçlar", "sonuç", "anlatır", "açıklayınız", "açıkla",
    "söyler", "misin", "midir", "mıdır", "yapılmıştır", "yapılır",
}


def _tr_lower(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def _keywords(query):

    tokens = re.findall(r"\w+", _tr_lower(query))

    return [t[:5] for t in tokens if len(t) >= 4 and t not in _STOPWORDS]


def search(query, client, data, top_k=TOP_K):

    response = client.generate_embedding(QUERY_PREFIX + query)

    query_vec = normalize(response.data[0].embedding)[0]

    cos_scores = data["embeddings"] @ query_vec

    # Anahtar kelime eşleşmesi: sorudaki önemli kelimelerin (kök olarak)
    # chunk içinde geçme oranı.
    if "lower" not in data:
        data["lower"] = [_tr_lower(c["text"]) for c in data["chunks"]]

    keys = _keywords(query)

    if keys:
        kw_scores = np.array(
            [sum(k in text for k in keys) / len(keys) for text in data["lower"]],
            dtype=np.float32,
        )
    else:
        kw_scores = np.zeros(len(cos_scores), dtype=np.float32)

    scores = cos_scores + KEYWORD_WEIGHT * kw_scores

    top_idx = np.argsort(-scores)[:top_k]

    results = [
        {
            "page": data["chunks"][i]["page"],
            "text": data["chunks"][i]["text"],
            "score": float(scores[i]),
            "cos": float(cos_scores[i]),
            "kw": float(kw_scores[i]),
            "has_keys": bool(keys),
        }
        for i in top_idx
    ]

    print(
        "[RAG] Sorgu:", query,
        "| Anahtarlar:", keys,
        "| Top:", [(r["page"], round(r["cos"], 3), round(r["kw"], 2)) for r in results],
    )

    return results


def is_relevant(r):
    """Embedding benzerliği eşiğin üstündeyse alakalı."""

    return r["cos"] >= MIN_COS


# ============================================================
# CEVAP
# ============================================================

def build_context(results):

    return "\n\n".join(
        f"[Sayfa {r['page']}]\n{r['text']}" for r in results
    )


def generate_answer(chat_client, query, results):

    system_message = (
        "Sen bir doküman soru-cevap asistanısın. "
        "YALNIZCA BAĞLAM bölümündeki bilgilere dayanarak cevap ver. "
        "Kendi genel bilgini kullanma, bağlamda olmayan hiçbir şey ekleme. "
        "Bağlamda soruyla ilgili herhangi bir bilgi varsa (kısmi olsa bile) "
        "onu kullanarak soruyu cevapla. "
        f"Bağlamda soruyla ilgili HİÇBİR bilgi yoksa SADECE şu cümleyi yaz: "
        f"'{PROMPT_NOT_FOUND}' "
        "SADECE sorulan şeyi cevapla: soru 'nedenleri' ise yalnızca nedenleri, "
        "'sonuçları' ise yalnızca sonuçları yaz; sorulmayan başlık (sonuçlar, "
        "önemi vb.) EKLEME. Bağlamdaki maddeleri olduğu gibi aktar, yeni madde "
        "üretme, yorum ekleme, konuyla ilgisiz başka olayların bilgisini karıştırma. "
        "Cevabı Türkçe, kısa ve net ver. Soruyu tekrar etme. "
        "Cevabın sonunda bilginin geçtiği sayfayı (Sayfa X) olarak belirt."
    )

    user_message = (
        f"BAĞLAM:\n\n{build_context(results)}\n\n"
        f"SORU: {query}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = chat_client.complete_chat(messages)

    if hasattr(response, "choices"):
        return response.choices[0].message.content

    if hasattr(response, "message"):
        return response.message.content

    if hasattr(response, "content"):
        return response.content

    return str(response)


def format_answer(text):
    """
    Modelin tek satırda yazdığı '• a • b • c' maddelerini alt alta
    Markdown listesine çevirir.
    """

    text = (text or "").strip()

    # Satır içi madde işaretlerini yeni satıra böl
    text = re.sub(r"\s*[•●▪◦]\s*", "\n- ", text)

    lines = []

    for line in text.split("\n"):

        line = line.rstrip()

        # "* madde" veya "– madde" -> "- madde" ("**kalın**" etkilenmez)
        m = re.match(r"^\s*[\*\-–]\s+(.*)", line)

        if m:
            line = "- " + m.group(1)

        lines.append(line)

    text = "\n".join(lines)

    # Listeden önce boş satır bırak (Markdown listesinin düzgün çıkması için)
    text = re.sub(r"([^\n])\n- ", r"\1\n\n- ", text, count=1)

    return text


_UNASKED_HEADER = re.compile(
    r"^\W*(sonuç\w*|önem\w*|etki\w*)\W*$", re.IGNORECASE
)
_UNASKED_STEMS = {"sonuç": "sonuç", "önem": "önem", "etki": "etki"}


def clean_answer(text, query):
    """
    1) Tekrar eden satırları siler (küçük modelin döngüsüne karşı).
    2) Soruda istenmeyen 'Sonuçlar/Önemi/Etkileri' bölümlerini keser.
    """

    q = _tr_lower(query)

    # --- Tekrarları sil ---
    seen = set()
    lines = []

    for line in text.split("\n"):

        if not line.strip():
            lines.append(line)
            continue

        key = re.sub(r"\W+", " ", _tr_lower(line)).replace(" kez ", " ").strip()

        if key in seen:
            continue

        seen.add(key)
        lines.append(line)

    # --- İstenmeyen bölümleri kes ---
    result = []

    for line in lines:

        header = _UNASKED_HEADER.match(_tr_lower(line.strip()))

        if header:

            stem = header.group(1)[:4]
            asked = any(stem == k[:4] and k in q for k in _UNASKED_STEMS)

            if not asked:

                # Başlık cevabın başındaysa sadece başlığı at, içeriği tut.
                # Daha önce içerik varsa buradan sonrasını tamamen kes.
                if any(r.strip() for r in result):
                    break

                continue

        result.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()

    return cleaned or text


def is_not_found(answer):

    low = _tr_lower(answer)

    return (
        "dokümanda bulunmuyor" in low
        or "dokümanda yer almamaktadır" in low
        or "dokümanda yer almıyor" in low
        or "yer almamaktadır" in low
    )


# ============================================================
# ARAYÜZ
# ============================================================

st.title("📚 Local RAG Assistant")

st.write("PDF dokümanınızı yükleyin ve doküman hakkında sorular sorun.")

st.divider()

st.subheader("📄 PDF Yükle")

uploaded_file = st.file_uploader("Bir PDF dosyası seçin", type=["pdf"])

if uploaded_file is None:
    st.info("Başlamak için yukarıdan bir PDF dosyası yükleyin.")
    st.stop()


# ============================================================
# PDF İŞLEME
# ============================================================

embedding_client, embedding_model_id = load_embedding_resources()

current_hash = calculate_file_hash(uploaded_file)

if st.session_state.file_hash != current_hash:

    cache_key = get_cache_key(current_hash, embedding_model_id)

    data = load_embedding_cache(cache_key)

    if data is not None:

        st.success("Bu PDF daha önce işlenmiş, önbellekten yüklendi.")

    else:

        with st.spinner("PDF metne dönüştürülüyor..."):
            pages = extract_text_from_pdf(uploaded_file)

        chunks = create_chunks(
            pages,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        if not chunks:
            st.error(
                "PDF'den metin çıkarılamadı. Taranmış (görüntü) bir PDF "
                "olabilir; bu durumda OCR gerekir."
            )
            st.stop()

        st.write(f"PDF {len(pages)} sayfa, {len(chunks)} chunk içeriyor.")

        embeddings = create_embeddings(chunks, embedding_client)

        save_embedding_cache(cache_key, chunks, embeddings)

        data = {"chunks": chunks, "embeddings": embeddings}

        st.success("PDF işlendi ve embedding'ler oluşturuldu!")

    # Hash, işlem başarıyla bittikten SONRA kaydedilir; hata olursa
    # bir sonraki çalıştırmada tekrar denenir.
    st.session_state.embedded_data = data
    st.session_state.processed_file = uploaded_file.name
    st.session_state.file_hash = current_hash


# ============================================================
# SORU-CEVAP
# ============================================================

st.divider()

if st.session_state.embedded_data is None:
    st.warning("Doküman işlenemedi.")
    st.stop()

data = st.session_state.embedded_data

st.subheader("💬 Soru Sor")

st.caption(
    f"Yüklü doküman: {st.session_state.processed_file} "
    f"({len(data['chunks'])} chunk)"
)

query = st.text_input("Sorunuzu yazın:", key="query_input")

ask_clicked = st.button("Sor", type="primary")

if ask_clicked and not query.strip():
    st.warning("Lütfen bir soru yazın.")

if ask_clicked and query.strip():

    with st.spinner("Dokümanda aranıyor..."):
        results = search(query, embedding_client, data, top_k=TOP_K)

    relevant = [r for r in results if is_relevant(r)]

    # Alakalı chunk yoksa LLM'e hiç gitme, kaynakça da gösterme.
    if not relevant:

        print("[RAG] Reddedildi (alakalı chunk yok).")

        st.markdown("### Cevap")
        st.write(NOT_FOUND_TEXT)

        st.stop()

    try:
        chat_client = load_chat_client()
    except Exception as e:
        st.error(f"Chat modeli hazırlanamadı: {e}")
        st.stop()

    with st.spinner("Cevap oluşturuluyor..."):
        answer = generate_answer(chat_client, query, relevant)

    st.markdown("### Cevap")

    # Model bilginin olmadığını söylediyse tek sabit cümle göster, kaynakça yok.
    if is_not_found(answer):
        print("[RAG] Model 'bilgi yok' dedi. Ham cevap:", answer)
        st.write(NOT_FOUND_TEXT)
        st.stop()

    st.markdown(clean_answer(format_answer(answer), query))

    st.markdown("### Kaynaklar")

    for i, result in enumerate(relevant, start=1):

        st.caption(
            f"Kaynak {i} — Sayfa {result['page']} "
            f"(Benzerlik: {result['score']:.3f})"
        )

        with st.expander(f"Kaynak {i} — içerik"):
            st.write(result["text"])