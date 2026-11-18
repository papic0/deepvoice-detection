"""전화망 열화: ffmpeg로 8kHz μ-law(G.711) 왕복 후 16kHz로 복원."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from vfp.common import TARGET_SR, load_audio, match_length, save_wav


def telephony_g711(y: np.ndarray) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg가 설치되어 있지 않음 (전화 음질 조건 생성 불가)")
    with tempfile.TemporaryDirectory() as td:
        src, mid, dst = (Path(td) / n for n in ("a.wav", "b.wav", "c.wav"))
        save_wav(src, y)
        q = ["-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(["ffmpeg", *q, "-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(mid)], check=True)
        subprocess.run(["ffmpeg", *q, "-i", str(mid), "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)], check=True)
        return match_length(load_audio(dst), len(y))


def add_g711_rows(rows: list[dict], out_dir: Path) -> list[dict]:
    """rows의 각 음성에 대해 '조건|g711' 사본을 만들고 새 행 목록을 반환 (이미 있으면 재사용)."""
    new = []
    for r in rows:
        path = Path(out_dir) / "g711" / r["condition"] / f"{r['utt_id']}.wav"
        if not path.exists():
            save_wav(path, telephony_g711(load_audio(r["path"])))
        new.append({**r, "condition": f"{r['condition']}|g711", "path": str(path)})
    return new
