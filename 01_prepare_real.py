"""1단계: 진짜 음성 준비.

진짜 발화 N개를 화자 균형을 맞춰 뽑아 16kHz mono wav로 저장하고 매니페스트를 만든다.

사용 예)
  # 로컬 폴더 (AI허브 .pcm, Zeroth 원본, LibriSpeech, 직접 녹음 등)
  python 01_prepare_real.py --source dir --input_dir D:/data/zeroth_korean/test_data_01 \
      --lang ko --out data/real_ko --n 200 --speaker_depth 2

  # Hugging Face 데이터셋 (인터넷 필요)
  python 01_prepare_real.py --source hf --hf_name Bingsu/zeroth-korean --hf_split test \
      --lang ko --out data/real_ko --n 200
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from vfp.common import TARGET_SR, load_audio, save_wav, write_manifest

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".pcm"}


def round_robin(groups: dict[str, list], n: int, seed: int) -> list:
    """화자별로 돌아가며 하나씩 뽑아 특정 화자 쏠림을 막는다."""
    rng = random.Random(seed)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    picked, i = [], 0
    while len(picked) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            picked.append(groups[k].pop())
        i += 1
    return picked


def from_dir(args) -> list[dict]:
    files = [p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT]
    if not files:
        raise SystemExit(f"오디오 파일을 찾지 못함: {args.input_dir}")
    groups = defaultdict(list)
    for p in files:
        spk = p.parents[args.speaker_depth - 1].name if args.speaker_depth > 0 else "unk"
        groups[spk].append(p)
    print(f"파일 {len(files)}개, 화자(폴더 기준) {len(groups)}명 발견")
    # 길이 필터 때문에 넉넉히 뽑은 뒤 걸러냄
    candidates = round_robin(groups, n=len(files), seed=args.seed)
    rows = []
    for p in candidates:
        if len(rows) >= args.n:
            break
        try:
            y = load_audio(p)
        except Exception as e:  # 깨진 파일은 건너뜀
            print(f"[건너뜀] {p}: {e}")
            continue
        dur = len(y) / TARGET_SR
        if not (args.min_sec <= dur <= args.max_sec):
            continue
        spk = p.parents[args.speaker_depth - 1].name if args.speaker_depth > 0 else "unk"
        utt = f"{args.lang}_{spk}_{p.stem}"
        rows.append(dict(utt_id=utt, y=y, speaker=spk, dur=dur))
    return rows


def from_hf(args) -> list[dict]:
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    from vfp.common import resample

    ds = load_dataset(args.hf_name, split=args.hf_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)  # 화자순 정렬된 데이터의 쏠림 완화
    # datasets 4.x부터 기본 디코더가 torchcodec으로 바뀌어 설치가 번거로움 → 원본 바이트를 직접 디코딩
    ds = ds.cast_column(args.hf_audio_col, Audio(decode=False))
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= args.n:
            break
        a = ex[args.hf_audio_col]
        src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
        y, sr = sf.read(src, dtype="float32", always_2d=True)
        y = resample(y.mean(axis=1), sr, TARGET_SR)
        dur = len(y) / TARGET_SR
        if not (args.min_sec <= dur <= args.max_sec):
            continue
        spk = str(ex.get(args.hf_speaker_col, "unk")) if args.hf_speaker_col else "unk"
        rows.append(dict(utt_id=f"{args.lang}_hf{i:06d}", y=y, speaker=spk, dur=dur))
    if args.hf_speaker_col is None:
        print("주의: 화자 정보 없이 앞에서부터 뽑음 (화자 균형 미보장)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dir", "hf"], required=True)
    ap.add_argument("--input_dir")
    ap.add_argument("--speaker_depth", type=int, default=1,
                    help="화자 이름이 들어 있는 상위 폴더 깊이 (1=바로 위 폴더, LibriSpeech/Zeroth 원본은 2)")
    ap.add_argument("--hf_name")
    ap.add_argument("--hf_split", default="test")
    ap.add_argument("--hf_audio_col", default="audio")
    ap.add_argument("--hf_speaker_col", default=None)
    ap.add_argument("--lang", required=True, help="ko / en 등")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = from_dir(args) if args.source == "dir" else from_hf(args)
    if len(rows) < args.n:
        print(f"주의: 요청 {args.n}개 중 {len(rows)}개만 확보")

    out = Path(args.out)
    manifest = []
    for r in rows:
        path = out / "wav" / f"{r['utt_id']}.wav"
        save_wav(path, r["y"])
        manifest.append(dict(utt_id=r["utt_id"], path=str(path), lang=args.lang, condition="real",
                             label=1, source_utt=r["utt_id"], speaker=r["speaker"],
                             dur_sec=f"{r['dur']:.2f}"))
    write_manifest(out / "manifest.csv", manifest)
    n_spk = len({m["speaker"] for m in manifest})
    total = sum(float(m["dur_sec"]) for m in manifest) / 60
    print(f"완료: {len(manifest)}개 발화, 화자 {n_spk}명, 총 {total:.1f}분 -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
