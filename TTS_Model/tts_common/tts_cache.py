# tts_common/tts_cache.py
import os
import hashlib
import shutil

CACHE_DIR_NAME = "tts_cache"

def make_cache_dir(base_dir=None):
    base = base_dir or os.path.join(os.getcwd(), "TTS_Model")
    cache_dir = os.path.join(base, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _hash_key(text: str, lang: str = "", desc: str = "", engine: str = "") -> str:
    key = (text or "") + "|" + (lang or "") + "|" + (desc or "") + "|" + (engine or "")
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return h

def cache_filepath(text: str, lang: str = "", desc: str = "", engine: str = "", base_dir=None, ext=".wav") -> str:
    cache_dir = make_cache_dir(base_dir)
    h = _hash_key(text, lang, desc, engine)
    filename = f"{engine or 'tts'}_{h}{ext}"
    return os.path.join(cache_dir, filename)

def exists_in_cache(text: str, lang: str = "", desc: str = "", engine: str = "", base_dir=None, ext=".wav") -> str | None:
    path = cache_filepath(text, lang, desc, engine, base_dir=base_dir, ext=ext)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    return None

def save_to_cache(src_path: str, text: str, lang: str = "", desc: str = "", engine: str = "", base_dir=None, ext=None) -> str:
    cache_dir = make_cache_dir(base_dir)
    _, s_ext = os.path.splitext(src_path)
    if ext is None:
        ext = s_ext or ".wav"
    dest = cache_filepath(text, lang, desc, engine, base_dir=base_dir, ext=ext)
    shutil.copyfile(src_path, dest)
    return dest