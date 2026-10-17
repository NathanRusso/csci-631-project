import io
import shutil
import time
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from kaggle.api.kaggle_api_extended import KaggleApi
from datasets import load_dataset
from tqdm import tqdm
import ast
import json

# Config
# Card dimensions (portrait: 500 width x 700 height)
TARGET_SIZE = (500, 700)
RESIZE_MODE = "pad"  # "pad" (black letterboxing) or "stretch"

BASE_OUTPUT_DIR = Path("../data")
RAW_TEMP_DIR = Path("./temp_kaggle_raw")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 1. Pokémon Dataset (Direct image files via Kaggle)
POKEMON_KAGGLE_SLUG = "ellimaaac/pokemon-tcg-all-image-cards"

# 2. Yu-Gi-Oh! Dataset (CSV pointing to YGOPRODeck images)
YUGIOH_KAGGLE_SLUG = "ioexception/yugioh-cards"
YUGIOH_WORKERS = 4              # Gentle concurrency pool
YUGIOH_MAX_REQ_PER_SEC = 10.0   # Strict limit: YGOPRODeck hard caps at 20 req/s

# 3. Magic: The Gathering Dataset (Streamed via Hugging Face)
MTG_HF_DATASET = "acidtib/tcg-mtg-cards"
MTG_MAX_SAMPLES = 20_000        # Caps sample count to save disk space


class RateLimiter:
    """Thread-safe rate limiter enforcing maximum requests per second."""
    def __init__(self, max_per_second: float):
        self.interval = 1.0 / max_per_second
        self.lock = threading.Lock()
        self.last_time = 0.0

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_time = time.time()


RATE_LIMITER = RateLimiter(max_per_second=YUGIOH_MAX_REQ_PER_SEC)


def create_safe_session() -> requests.Session:
    """Configures HTTP session with polite headers and exponential retry handling."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://ygoprodeck.com/",
    })
    
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def normalize_and_resize(img: Image.Image, size: tuple[int, int], mode: str = "pad") -> Image.Image:
    """Resizes an image into uniform CNN input tensor dimensions."""
    target_w, target_h = size
    if mode == "stretch":
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)

    resized_content = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_content, (paste_x, paste_y))
    return canvas


def fetch_and_save_yugioh_url(url: str, output_path: Path, target_size: tuple[int, int], session: requests.Session) -> bool:
    """Downloads a single YGOPRODeck image respecting site rate limits."""
    if not isinstance(url, str) or not url.startswith("http"):
        return False

    # Block until within the max_per_second limit
    RATE_LIMITER.wait()

    try:
        response = session.get(url, timeout=15)

        # Handle rate limit hit gracefully
        if response.status_code == 429:
            print("\n[Warning] Hit 429 from YGOPRODeck! Pausing for 65 seconds to clear ban...")
            time.sleep(65)
            response = session.get(url, timeout=15)

        response.raise_for_status()

        with Image.open(io.BytesIO(response.content)) as img:
            img = img.convert("RGB")
            resized = normalize_and_resize(img, target_size, mode=RESIZE_MODE)
            resized.save(output_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def process_pokemon(base_output: Path, target_size: tuple[int, int], api: KaggleApi):
    """Downloads Pokémon cards archive and resizes directly."""
    out_dir = base_output / "pokemon"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = RAW_TEMP_DIR / "pokemon"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Downloading Pokémon dataset: '{POKEMON_KAGGLE_SLUG}'...")
    api.dataset_download_files(POKEMON_KAGGLE_SLUG, path=temp_dir, unzip=True)

    files = [p for p in temp_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]
    print(f"Resizing {len(files)} Pokémon images to {target_size[0]}x{target_size[1]}...")

    for idx, path in enumerate(tqdm(files, desc="Pokemon Cards"), start=1):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                resized = normalize_and_resize(img, target_size, mode=RESIZE_MODE)
                resized.save(out_dir / f"pokemon_{idx:06d}.jpg", "JPEG", quality=92)
        except Exception:
            continue

    shutil.rmtree(temp_dir, ignore_errors=True)


def extract_yugioh_urls(df: pd.DataFrame) -> list[str]:
    """Extracts valid YGOPRODeck image URLs regardless of how Kaggle formatted them."""
    # 1. Direct image_url column
    for col in ["image_url", "image_url_small", "url"]:
        if col in df.columns:
            return df[col].dropna().drop_duplicates().tolist()

    # 2. Nested 'card_images' column (standard YGOPRODeck JSON export)
    if "card_images" in df.columns:
        print("Parsing image URLs from nested 'card_images' column...")
        urls = []
        for entry in df["card_images"].dropna():
            if not isinstance(entry, str):
                continue
            try:
                # Can be valid JSON or Python stringified dict
                data = json.loads(entry.replace("'", '"')) if '"' not in entry else json.loads(entry)
            except Exception:
                try:
                    data = ast.literal_eval(entry)
                except Exception:
                    continue

            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                url = data[0].get("image_url") or data[0].get("image_url_small")
                if url:
                    urls.append(url)
            elif isinstance(data, dict):
                url = data.get("image_url") or data.get("image_url_small")
                if url:
                    urls.append(url)

        if urls:
            return list(dict.fromkeys(urls))  # remove duplicates preserving order

    # 3. Fallback: Build URLs directly from card IDs
    id_cols = [c for c in ["id", "card_id", "konami_id"] if c in df.columns]
    if id_cols:
        col = id_cols[0]
        print(f"Building image URLs directly from '{col}' column...")
        card_ids = df[col].dropna().astype(str).str.strip().tolist()
        card_ids = [cid.split(".")[0] for cid in card_ids if cid.replace(".", "").isdigit()]
        return [f"https://images.ygoprodeck.com/images/cards/{cid}.jpg" for cid in dict.fromkeys(card_ids)]

    raise ValueError(f"Could not extract URLs. Available columns: {list(df.columns)}")


def process_yugioh(base_output: Path, target_size: tuple[int, int], api: KaggleApi):
    """Downloads Yu-Gi-Oh CSV and extracts image URLs safely."""
    out_dir = base_output / "yugioh"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = RAW_TEMP_DIR / "yugioh"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Downloading Yu-Gi-Oh dataset: '{YUGIOH_KAGGLE_SLUG}'...")
    api.dataset_download_files(YUGIOH_KAGGLE_SLUG, path=temp_dir, unzip=True)

    csv_files = list(temp_dir.rglob("*.csv"))
    if not csv_files:
        print("Error: Could not locate CSV in Yu-Gi-Oh archive.")
        return

    csv_file = sorted(csv_files, key=lambda p: "cards" in p.name.lower(), reverse=True)[0]
    print(f"Reading '{csv_file.name}'...")
    df = pd.read_csv(csv_file)

    urls = extract_yugioh_urls(df)
    print(f"Discovered {len(urls)} valid Yu-Gi-Oh card URLs. Starting throttled download...")

    session = create_safe_session()

    with ThreadPoolExecutor(max_workers=YUGIOH_WORKERS) as executor:
        futures = {
            executor.submit(
                fetch_and_save_yugioh_url,
                url,
                out_dir / f"yugioh_{idx:06d}.jpg",
                target_size,
                session
            ): idx
            for idx, url in enumerate(urls, start=1)
        }

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Yu-Gi-Oh Cards"):
            pass

    shutil.rmtree(temp_dir, ignore_errors=True)


def process_mtg_streaming(base_output: Path, target_size: tuple[int, int], max_samples: int):
    """Streams Magic cards directly from Hugging Face into memory (capped at max_samples)."""
    out_dir = base_output / "mtg"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/3] Streaming Hugging Face dataset '{MTG_HF_DATASET}' (capped at {max_samples:,} images)...")
    ds = load_dataset(MTG_HF_DATASET, split="train", streaming=True)

    saved_count = 0
    with tqdm(total=max_samples, desc="MTG Streaming") as pbar:
        for item in ds:
            if saved_count >= max_samples:
                break

            raw_img = item.get("image")
            if raw_img is None:
                continue

            try:
                if isinstance(raw_img, Image.Image):
                    img = raw_img
                elif isinstance(raw_img, dict) and "bytes" in raw_img:
                    img = Image.open(io.BytesIO(raw_img["bytes"]))
                else:
                    continue

                img = img.convert("RGB")
                resized = normalize_and_resize(img, target_size, mode=RESIZE_MODE)
                saved_count += 1
                resized.save(out_dir / f"mtg_{saved_count:06d}.jpg", "JPEG", quality=92)
                pbar.update(1)
            except Exception:
                continue


def main():
    # Setup environment variables if they exists
    load_dotenv("../.env")   

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Authenticating Kaggle API...")
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    # Step 1: Pokémon cards
    process_pokemon(BASE_OUTPUT_DIR, TARGET_SIZE, kaggle_api)

    # Step 2: Yu-Gi-Oh cards (with YGOPRODeck rate protection)
    process_yugioh(BASE_OUTPUT_DIR, TARGET_SIZE, kaggle_api)

    # Step 3: MTG cards (streaming capped to 20,000)
    process_mtg_streaming(BASE_OUTPUT_DIR, TARGET_SIZE, MTG_MAX_SAMPLES)

    # Final cleanup of any lingering raw files
    shutil.rmtree(RAW_TEMP_DIR, ignore_errors=True)
    print(f"\nAll tasks complete! Images saved to: {BASE_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()