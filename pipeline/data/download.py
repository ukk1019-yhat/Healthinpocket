import zipfile
import shutil
from pathlib import Path
from typing import Optional


def download_kaggle(dataset: str, dest: Path) -> Path:
    try:
        import kagglehub
    except ImportError:
        raise ImportError("Install kagglehub: pip install kagglehub")

    path = kagglehub.dataset_download(dataset)
    src = Path(path)

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def download_from_url(url: str, dest: Path) -> Path:
    import requests
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    local_path = dest / url.split("/")[-1]

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(local_path, "wb") as f, tqdm(
        desc=local_path.name, total=total, unit="B", unit_scale=True
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    if local_path.suffix == ".zip":
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(dest)
        local_path.unlink()

    return dest


def prepare_data_source(cfg: dict) -> Path:
    data_cfg = cfg["data"]
    source = data_cfg["source"]
    dest = Path(data_cfg["local_path"])

    if source == "kaggle":
        dataset = data_cfg["kaggle_dataset"]
        print(f"Downloading Kaggle dataset: {dataset}")
        download_kaggle(dataset, dest)
    elif source == "url":
        url = data_cfg.get("url")
        if url:
            print(f"Downloading from URL: {url}")
            download_from_url(url, dest)
    else:
        print(f"Using local data at {dest}")
        if not dest.exists():
            raise FileNotFoundError(f"Local data path not found: {dest}")

    return dest
