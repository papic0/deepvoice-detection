"""탐지기 가중치와 검증용 예제 데이터 내려받기.

  python setup_weights.py                 # 전부
  python setup_weights.py --only aasist   # 하나만 (aasist / ssl_aasist / antideepfake / toy)

저장 위치 (기본 weights/)
  weights/aasist/           clovaai/aasist 저장소 (가중치 포함)                약 10MB
  weights/ssl_aasist/       Wav2Vec2-AASIST 학습 가중치 (Google Drive)          파일당 약 1.2GB
  weights/antideepfake/     AntiDeepfake MMS-300M (Zenodo, 실패 시 Hugging Face)  약 1.2GB
  data/toy_example/         ASVspoof 2019 LA 일부 150개 (AntiDeepfake 원저자 검증용)  약 20MB

자동 다운로드가 막힌 경우(학교망 차단, Google Drive 일일 한도 등) 화면에 나오는 주소에서
직접 받아 위 폴더에 넣으면 됨. 파일 이름은 그대로 두는 것을 권장.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SSL_DRIVE = "https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB"
ADF_ZENODO = "https://zenodo.org/records/15580543/files/mms_300m.ckpt?download=1"
ADF_HF_REPO = "nii-yamagishilab/mms-300m-anti-deepfake"
TOY_URL = "https://zenodo.org/records/7497769/files/project-04-toy_example.tar.gz?download=1"


def say(msg):
    print(msg, flush=True)


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "deepvoice-bench"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.replace(dst)


def get_aasist(root: Path) -> bool:
    dst = root / "aasist"
    if (dst / "models" / "weights" / "AASIST.pth").exists():
        say("[aasist] 이미 있음")
        return True
    if shutil.which("git"):
        say("[aasist] git clone https://github.com/clovaai/aasist")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/clovaai/aasist", str(dst)], check=True)
            return True
        except Exception as e:
            say(f"  git 실패: {e} → ZIP으로 재시도")
    say("[aasist] GitHub ZIP 내려받기")
    try:
        import zipfile

        z = root / "aasist.zip"
        download("https://codeload.github.com/clovaai/aasist/zip/refs/heads/main", z)
        with zipfile.ZipFile(z) as zf:
            top = zf.namelist()[0].split("/")[0]
            for n in zf.namelist():
                if not (root / n).resolve().is_relative_to(root.resolve()):
                    raise RuntimeError(f"잘못된 압축 경로: {n}")
            zf.extractall(root)
        z.unlink()
        shutil.move(str(root / top), str(dst))
        return (dst / "models" / "weights" / "AASIST.pth").exists()
    except Exception as e:
        say(f"  실패: {e}\n  → https://github.com/clovaai/aasist 에서 ZIP으로 받아 {dst} 에 풀 것")
        return False


def get_ssl_aasist(root: Path) -> bool:
    dst = root / "ssl_aasist"
    if list(dst.rglob("*.pth")):
        say(f"[ssl_aasist] 이미 있음: {[p.name for p in dst.rglob('*.pth')]}")
        return True
    say("[ssl_aasist] Google Drive 폴더에서 내려받기 (gdown)")
    try:
        import gdown

        gdown.download_folder(SSL_DRIVE, output=str(dst), quiet=False, use_cookies=False)
    except Exception as e:
        say(f"  자동 다운로드 실패: {e}")
    if list(dst.rglob("*.pth")):
        return True
    say(f"  → 브라우저로 {SSL_DRIVE} 에서 LA_model.pth (필수), Best_LA_model_for_DF.pth (선택)를 받아 {dst} 에 넣을 것")
    return False


def get_antideepfake(root: Path) -> bool:
    dst = root / "antideepfake"
    for pat in ("*.ckpt", "*.safetensors"):
        if list(dst.rglob(pat)):
            say("[antideepfake] 이미 있음")
            return True
    say("[antideepfake] Zenodo에서 mms_300m.ckpt 내려받기")
    try:
        download(ADF_ZENODO, dst / "mms_300m.ckpt")
        return True
    except Exception as e:
        say(f"  Zenodo 실패: {e} → Hugging Face 시도")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(ADF_HF_REPO, local_dir=str(dst), allow_patterns=["*.safetensors", "*.json", "*.py", "*.md"])
        if list(dst.rglob("*.safetensors")):
            return True
    except Exception as e:
        say(f"  Hugging Face 실패: {e}")
    say(f"  → 직접 받기: {ADF_ZENODO.split('?')[0]} 를 {dst} 에 저장")
    return False


def get_toy(data_root: Path) -> bool:
    dst = data_root / "toy_example"
    if (dst / "protocol.txt").exists():
        say("[toy] 이미 있음")
        return True
    say("[toy] 원저자 검증용 예제(ASVspoof 2019 LA 일부) 내려받기")
    tgz = data_root / "toy_example.tar.gz"
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        download(TOY_URL, tgz)
        with tarfile.open(tgz) as t:
            # 압축 안의 경로가 data_root 밖으로 나가지 않는지 확인 후 해제
            for m in t.getmembers():
                if not (data_root / m.name).resolve().is_relative_to(data_root.resolve()):
                    raise RuntimeError(f"잘못된 압축 경로: {m.name}")
            t.extractall(data_root)
        tgz.unlink()
        found = list(data_root.rglob("protocol.txt"))
        if found and found[0].parent != dst:
            shutil.move(str(found[0].parent), str(dst))
        return (dst / "protocol.txt").exists()
    except Exception as e:
        say(f"  실패: {e}\n  → 직접 받기: {TOY_URL.split('?')[0]} 를 {data_root} 에 풀 것")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--only", choices=["aasist", "ssl_aasist", "antideepfake", "toy"])
    a = ap.parse_args()
    w, d = Path(a.weights_dir), Path(a.data_dir)
    jobs = {"aasist": lambda: get_aasist(w), "ssl_aasist": lambda: get_ssl_aasist(w),
            "antideepfake": lambda: get_antideepfake(w), "toy": lambda: get_toy(d)}
    todo = [a.only] if a.only else list(jobs)
    ok = {k: jobs[k]() for k in todo}
    say("\n결과: " + ", ".join(f"{k} {'OK' if v else '실패'}" for k, v in ok.items()))
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
