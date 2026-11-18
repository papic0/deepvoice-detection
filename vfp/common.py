"""공통 유틸: 오디오 입출력, 매니페스트(csv) 처리."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000  # AASIST 입력 샘플레이트

MANIFEST_FIELDS = [
    "utt_id",      # 조건 내에서 고유한 발화 ID
    "path",        # 오디오 파일 경로
    "lang",        # ko / en / asv19 등
    "condition",   # real 또는 생성 시스템 이름(dac_16k 등), 전화 열화 시 '|g711' 접미사
    "label",       # 1 = 진짜(bona fide), 0 = 가짜(spoof)
    "source_utt",  # 재합성의 원본 발화 ID (real이면 자기 자신)
    "speaker",
    "dur_sec",
]


def load_audio(path: str | os.PathLike, sr: int = TARGET_SR) -> np.ndarray:
    """mono float32, 지정 샘플레이트로 로드. .pcm은 16kHz 16bit little-endian(AI허브 형식)으로 간주."""
    path = str(path)
    if path.lower().endswith(".pcm"):
        y = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768.0
        file_sr = 16000
    else:
        try:
            y, file_sr = sf.read(path, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
        except Exception:
            import librosa  # mp3/m4a 등 soundfile이 못 읽는 형식

            y, file_sr = librosa.load(path, sr=None, mono=True)
            y = y.astype(np.float32)
    if file_sr != sr:
        y = resample(y, file_sr, sr)
    return y.astype(np.float32)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32)
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    return AF.resample(t, sr_in, sr_out).numpy().astype(np.float32)


def save_wav(path: str | os.PathLike, y: np.ndarray, sr: int = TARGET_SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")


def match_length(y: np.ndarray, n: int) -> np.ndarray:
    """재합성 결과 길이를 원본과 맞춤(코덱 패딩으로 길이가 달라지는 것 보정)."""
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def read_manifest(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: str | os.PathLike, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():  # NVIDIA CUDA 및 AMD ROCm 빌드 모두 'cuda'로 잡힘
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
