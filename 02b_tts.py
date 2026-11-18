"""2단계(TTS): 공개 음성 복제 TTS로 한국어 가짜 음성 생성.

각 진짜 발화마다 '같은 화자의 다른 발화'를 참조 음성으로 주고, '같은 문장'을 말하게 함
→ 진짜와 가짜가 화자·문장이 같고 생성 여부만 다름 (01_prepare_ko.py의 tts_meta.csv 사용)

시스템 (서로 패키지 버전이 충돌하므로 각자 별도 가상환경에서 실행)
  xtts       : Coqui XTTS v2 (2023) — GPT + HiFi-GAN 계열, 가중치 CPML(비상업)
               → 다국어 딥페이크 데이터셋(MLAAD 등)에 자주 쓰인 '오래된 세대'
  chatterbox : Resemble AI Chatterbox Multilingual (2025.9) — 음성 토큰 LM + flow matching, MIT
               (출력에 Resemble의 비가청 워터마크(Perth)가 기본 삽입됨)
  omnivoice  : k2-fsa OmniVoice (2026.3) — 확산 언어모델형, Apache-2.0, 한국어 학습 약 8,600시간
               → 비교 대상 탐지기 3종이 모두 공개된 뒤 나온 '처음 보는 세대'

  python 02b_tts.py --system chatterbox --meta data/real_ko/tts_meta.csv --out data/fake_ko_tts --telephony g711
  (이미 만든 파일은 건너뛰므로 중단 후 다시 실행해도 이어서 진행)
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vfp.common import TARGET_SR, load_audio, pick_device, resample, save_wav, write_manifest


# ------------------------------------------------------------------ 시스템별 로더: f(text, ref_path, ref_text) -> (wav, sr)
def load_xtts(device):
    os.environ.setdefault("COQUI_TOS_AGREED", "1")  # XTTS 가중치 라이선스(CPML, 비상업) 동의 → README 참고
    import TTS.tts.models.xtts as xm

    def _load(audiopath, sampling_rate):  # torchaudio.load(=torchcodec 필요) 대신 soundfile 사용
        y = load_audio(audiopath, sampling_rate)
        return torch.from_numpy(y)[None].clamp_(-1, 1)

    xm.load_audio = _load
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sr = tts.synthesizer.output_sample_rate

    def f(text, ref, ref_text):
        return np.asarray(tts.tts(text=text, speaker_wav=ref, language="ko"), dtype=np.float32), sr

    return f


def load_chatterbox(device):
    import chatterbox.models.tokenizers.tokenizer as cbt
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # 중국어 형태소 분석기(pkuseg) 모델을 GitHub에서 받다가 시간 초과가 남 → 한국어에는 불필요하므로 생략
    cbt.ChineseCangjieConverter._init_segmenter = lambda self: setattr(self, "segmenter", None)

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    def f(text, ref, ref_text):
        wav = model.generate(text, language_id="ko", audio_prompt_path=ref)
        return wav.squeeze().float().cpu().numpy(), model.sr

    return f


def load_omnivoice(device):
    from omnivoice import OmniVoice

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)

    def f(text, ref, ref_text):
        audio = model.generate(text=text, language="ko", ref_audio=ref, ref_text=ref_text or None)
        return np.asarray(audio[0], dtype=np.float32), model.sampling_rate

    return f


SYSTEMS = {"xtts": load_xtts, "chatterbox": load_chatterbox, "omnivoice": load_omnivoice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--meta", default="data/real_ko/tts_meta.csv")
    ap.add_argument("--out", default="data/fake_ko_tts")
    ap.add_argument("--telephony", choices=["none", "g711"], default="g711")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = pick_device(a.device)
    with open(a.meta, encoding="utf-8") as fh:
        meta = list(csv.DictReader(fh))
    if a.limit:
        meta = meta[: a.limit]
    out = Path(a.out)
    print(f"[{a.system}] 장치 {device}, 대상 {len(meta)}개")

    t0 = time.time()
    gen = SYSTEMS[a.system](device)
    print(f"[{a.system}] 모델 불러오기 {time.time() - t0:.0f}초")

    rows, n_new, n_bad = [], 0, 0
    t1 = time.time()
    for i, m in enumerate(meta):
        path = out / a.system / f"{m['utt_id']}.wav"
        if not path.exists():
            try:
                torch.manual_seed(i)
                wav, sr = gen(m["text"], m["ref_path"], m.get("ref_text", ""))
                z = resample(wav, sr, TARGET_SR)
            except Exception:
                print(f"[오류] {m['utt_id']}\n{traceback.format_exc(limit=3)}")
                n_bad += 1
                continue
            dur, real_dur = len(z) / TARGET_SR, float(m["dur_sec"])
            if not np.isfinite(z).all() or np.abs(z).max() < 1e-3 or dur < 0.5:
                print(f"[제외] {m['utt_id']}: 무음·비정상 출력")
                n_bad += 1
                continue
            if not (0.4 <= dur / real_dur <= 2.5):
                print(f"[주의] {m['utt_id']}: 길이 {dur:.1f}초 (진짜 {real_dur:.1f}초) — 문장 누락·반복 가능성")
            save_wav(path, z / max(1.0, float(np.abs(z).max())))
            n_new += 1
            if n_new % 20 == 0:
                print(f"  {i + 1}/{len(meta)}  ({(time.time() - t1) / n_new:.1f}초/개)")
        info = sf.info(str(path))
        rows.append(dict(utt_id=m["utt_id"], path=str(path), lang="ko", condition=a.system, label=0,
                         source_utt=m["utt_id"], speaker=m["speaker"], dur_sec=f"{info.duration:.2f}"))
    print(f"[{a.system}] 생성 {n_new}개 (기존 {len(rows) - n_new}개 재사용), 실패 {n_bad}개, {time.time() - t1:.0f}초")

    if a.telephony == "g711":
        from vfp.telephony import add_g711_rows

        print(f"[{a.system}] 전화 음질(G.711) 사본 생성 중...")
        rows += add_g711_rows([r for r in rows if r["condition"] == a.system], out)
    write_manifest(out / a.system / "manifest.csv", rows)
    print(f"완료 -> {out / a.system / 'manifest.csv'}")


if __name__ == "__main__":
    main()
