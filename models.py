from foundry_local_sdk import Configuration, FoundryLocalManager


# ============================================================
# MODEL AYARLARI (değiştirmek için sadece buraya bak)
# ============================================================

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# 1.5B model Türkçede ve "sadece bağlama dayan" talimatında zayıf kalır.
# Bilgisayarın yetiyorsa "qwen2.5-3b" veya "qwen2.5-7b" dene
# (Foundry Local kataloğunda varsa).
CHAT_MODEL_ALIAS = "qwen2.5-1.5b"

EMBEDDING_PRIORITIES = ["cuda-gpu", "openvino-gpu", "generic-gpu", "generic-cpu"]
CHAT_PRIORITIES = ["openvino-gpu", "cuda-gpu", "generic-gpu", "generic-cpu"]


def get_manager():

    config = Configuration(app_name="Local-RAG-Assistant")

    try:
        FoundryLocalManager.initialize(config)
        print("Foundry Local bağlantısı başarılı!")

    except Exception as e:

        if "already been initialized" in str(e):
            print("Foundry Local zaten başlatılmış.")
        else:
            raise

    manager = FoundryLocalManager.instance

    if manager is None:
        raise RuntimeError("Foundry Local Manager oluşturulamadı.")

    return manager


def _select_best_variant(model, priorities):
    """
    Öncelik sırasına göre cache'lenmiş ilk uygun varyantı döndürür.
    """

    for priority in priorities:

        for variant in model.variants:

            if priority in variant.id.lower() and variant.is_cached:
                return variant

    return None


def _get_model(manager, alias, priorities, label):

    if manager is None:
        raise RuntimeError(f"{label} modeli için manager bulunamadı.")

    model = manager.catalog.get_model(alias)

    if model is None:
        raise RuntimeError(f"{alias} modeli katalogda bulunamadı.")

    variant = _select_best_variant(model, priorities)

    if variant is not None:
        model.select_variant(variant)
        print(f"{label} varyantı seçildi: {variant.id}")
    else:
        print(f"{label}: cache'lenmiş uygun varyant bulunamadı, varsayılan kullanılacak.")

    print(f"FINAL {label.upper()} MODEL: {model.id}")

    return model


def get_embedding_model(manager):
    return _get_model(manager, EMBEDDING_MODEL_ALIAS, EMBEDDING_PRIORITIES, "Embedding")


def get_chat_model(manager):
    return _get_model(manager, CHAT_MODEL_ALIAS, CHAT_PRIORITIES, "Chat")