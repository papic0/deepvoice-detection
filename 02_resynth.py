"""2단계: 가짜(재합성) 음성 생성.

진짜 음성을 각 생성 시스템에 '통과'시켜 같은 문장·같은 화자의 가짜 음성을 만든다.
(남의 목소리를 새로 복제하는 것이 아니라, 원본을 코덱/보코더로 다시 합성하는 copy-synthesis 방식)

시스템 구분
  [대조군: 보코더/고전 방식 — 기존 탐지기가 학습한 계열]
    griffinlim   : 신경망 없는 고전 위상복원 (오프라인 동작)
    hifigan_16k  : HiFi-GAN 신경망 보코더 (speechbrain)
  [실험군: 신경망 코덱 — 최신 LLM형 TTS가 쓰는 계열]
    dac_16k      : Descript Audio Codec 16kHz, 전체 코드북
    dac_16k_q4   : 같은 코덱, 코드북 4개만 사용 (저비트레이트)
    encodec_6    : Meta EnCodec 24kHz, 6kbps
    encodec_1.5  : Meta EnCodec 24kHz, 1.5kbps
    mimi_q8      : Kyutai Mimi (Moshi 음성 LLM의 코덱), 코드북 8개
    snac_24k     : SNAC 24kHz (Orpheus 등 LLM TTS가 사용)

전화망 열화(--telephony g711): 진짜·가짜 모두 8kHz μ-law로 왕복시켜 '|g711' 조건을 추가 생성.

사용 예)
  python 02_resynth.py --real data/real_ko/manifest.csv --out data/fake_ko \
      --systems griffinlim,hifigan_16k,dac_16k,dac_16k_q4,encodec_6,encodec_1.5,mimi_q8,snac_24k \
      --telephony g711
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from vfp.common import (TARGET_SR, load_audio, match_length, pick_device, read_manifest,
                        resample, save_wav, write_manifest)

# ---------------------------------------------------------------- 생성 시스템 정의
# 각 로더는 (내부 샘플레이트, 함수 f(y: np.ndarray@sr) -> np.ndarray@sr)를 반환


def load_griffinlim(device):
    import librosa

    def f(y):
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        return librosa.griffinlim(S, n_iter=60, hop_length=256, n_fft=1024, length=len(y))

    return 16000, f


def load_hifigan_16k(device):
    from speechbrain.inference.vocoders import HIFIGAN
    try:  # speechbrain 1.x
        from speechbrain.lobes.models.HifiGAN import mel_spectrogram
    except ImportError:  # 0.5.x (철자 다름)
        from speechbrain.lobes.models.HifiGAN import mel_spectogram as mel_spectrogram

    kw = {}
    try:  # Windows: 기본 '심볼릭 링크' 방식은 관리자 권한이 필요함 → 파일 복사 방식 사용
        from speechbrain.utils.fetching import LocalStrategy

        kw["local_strategy"] = LocalStrategy.COPY
    except ImportError:
        pass
    hifi = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                savedir="pretrained/hifigan_16k", run_opts={"device": device}, **kw)

    def f(y):
        x = torch.from_numpy(y).to(device)
        # speechbrain 모델 카드의 16kHz LibriTTS 보코더 멜 설정과 동일
        mel = mel_spectrogram(audio=x, sample_rate=16000, hop_length=256, win_length=1024,
                              n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                              normalized=False, norm="slaney", mel_scale="slaney",
                              compression=True)
        with torch.no_grad():
            wav = hifi.decode_batch(mel.unsqueeze(0))
        return wav.squeeze().cpu().numpy()

    return 16000, f


def _load_dac(device, n_q):
    import dac

    import time

    for attempt in range(4):  # GitHub 릴리스 다운로드가 가끔 504로 실패 → 재시도
        try:
            path = dac.utils.download(model_type="16khz")
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  DAC 가중치 다운로드 재시도 ({attempt + 1}/3): {e}")
            time.sleep(10 * (attempt + 1))
    model = dac.DAC.load(path).to(device).eval()

    def f(y):
        x = torch.from_numpy(y)[None, None].to(device)
        with torch.no_grad():
            x = model.preprocess(x, 16000)
            z, *_ = model.encode(x, n_quantizers=n_q)
            out = model.decode(z)
        return out.squeeze().cpu().numpy()

    return 16000, f


def load_dac_16k(device):
    return _load_dac(device, None)


def load_dac_16k_q4(device):
    return _load_dac(device, 4)


def _load_encodec(device, bw):
    from transformers import EncodecModel

    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()

    def f(y):
        x = torch.from_numpy(y)[None, None].to(device)
        with torch.no_grad():
            enc = model.encode(x, bandwidth=bw)
            out = model.decode(enc.audio_codes, enc.audio_scales)[0]
        return out.squeeze().cpu().numpy()

    return 24000, f


def load_encodec_6(device):
    return _load_encodec(device, 6.0)


def load_encodec_1_5(device):
    return _load_encodec(device, 1.5)


def load_mimi_q8(device):
    from transformers import MimiModel

    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    def f(y):
        x = torch.from_numpy(y)[None, None].to(device)
        with torch.no_grad():
            codes = model.encode(x, num_quantizers=8).audio_codes
            out = model.decode(codes)[0]
        return out.squeeze().cpu().numpy()

    return 24000, f


def load_snac_24k(device):
    from snac import SNAC

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    def f(y):
        x = torch.from_numpy(y)[None, None].to(device)
        with torch.no_grad():
            out = model.decode(model.encode(x))
        return out.squeeze().cpu().numpy()

    return 24000, f


SYSTEMS = {
    "griffinlim": load_griffinlim,
    "hifigan_16k": load_hifigan_16k,
    "dac_16k": load_dac_16k,
    "dac_16k_q4": load_dac_16k_q4,
    "encodec_6": load_encodec_6,
    "encodec_1.5": load_encodec_1_5,
    "mimi_q8": load_mimi_q8,
    "snac_24k": load_snac_24k,
}

# ---------------------------------------------------------------- 전화망 열화


def telephony_g711(y: np.ndarray) -> np.ndarray:
    """ffmpeg로 8kHz μ-law(G.711) 왕복 후 16kHz로 복원."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg가 설치되어 있지 않음 (--telephony 사용 불가)")
    with tempfile.TemporaryDirectory() as td:
        src, mid, dst = (Path(td) / n for n in ("a.wav", "b.wav", "c.wav"))
        save_wav(src, y)
        q = ["-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(["ffmpeg", *q, "-i", src, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", mid], check=True)
        subprocess.run(["ffmpeg", *q, "-i", mid, "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", dst], check=True)
        return match_length(load_audio(dst), len(y))


# ---------------------------------------------------------------- 실행


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="01단계 매니페스트")
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--telephony", choices=["none", "g711"], default="none")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: 앞에서 N개만 처리")
    ap.add_argument("--no_real_telephony", action="store_true",
                    help="진짜 음성의 전화 음질 사본은 만들지 않음 (01_prepare_ko.py가 이미 만든 경우)")
    args = ap.parse_args()

    device = pick_device(args.device)
    reals = [r for r in read_manifest(args.real) if r["condition"] == "real"]
    if args.limit:
        reals = reals[: args.limit]
    out = Path(args.out)
    print(f"장치: {device}, 원본 {len(reals)}개")

    rows, failed = [], []
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        if name not in SYSTEMS:
            raise SystemExit(f"알 수 없는 시스템: {name} (가능: {', '.join(SYSTEMS)})")
        t0 = time.time()
        try:
            sr, fn = SYSTEMS[name](device)
        except Exception as e:
            print(f"[실패] {name} 로딩 불가 -> 건너뜀: {e}")
            failed.append(name)
            continue
        n_ok = 0
        for r in reals:
            y = load_audio(r["path"])
            try:
                z = resample(fn(resample(y, TARGET_SR, sr)), sr, TARGET_SR)
            except Exception:
                print(f"[오류] {name} / {r['utt_id']}\n{traceback.format_exc(limit=2)}")
                continue
            z = match_length(z, len(y))
            if not np.isfinite(z).all() or np.abs(z).max() < 1e-4:
                print(f"[경고] {name} / {r['utt_id']}: 출력이 비정상(무음/NaN) -> 제외")
                continue
            path = out / name / f"{r['utt_id']}.wav"
            save_wav(path, z)
            rows.append(dict(utt_id=r["utt_id"], path=str(path), lang=r["lang"], condition=name,
                             label=0, source_utt=r["source_utt"], speaker=r["speaker"],
                             dur_sec=r["dur_sec"]))
            n_ok += 1
        print(f"{name}: {n_ok}/{len(reals)}개 생성, {time.time() - t0:.0f}초")
        del fn
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.telephony == "g711":
        print("전화망(G.711) 열화 조건 생성 중...")
        from vfp.telephony import add_g711_rows

        base = ([] if args.no_real_telephony else [dict(r) for r in reals]) + [dict(r) for r in rows]
        rows += add_g711_rows(base, out)

    write_manifest(out / "manifest.csv", rows)
    print(f"완료: {len(rows)}개 -> {out / 'manifest.csv'}")
    if failed:
        print(f"주의: 로딩 실패한 시스템 = {failed} (패키지 설치/인터넷 확인)")


if __name__ == "__main__":
    main()
