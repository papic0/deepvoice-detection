"""1단계(한국어): 공개 한국어 진짜 음성 + TTS 복제용 참조 음성 준비.

데이터: Zeroth-Korean 평가(test) 세트 — 457개 발화, 화자 10명, CC BY 4.0, 낭독체
  - AntiDeepfake 학습 데이터 목록(protocols/README.md)에 없음 → 데이터 중복 없이 공정 비교 가능
    (참고: FLEURS 한국어, AI Hub 기반 DSD는 AntiDeepfake 학습에 쓰였으므로 평가용 진짜 음성으로 쓰지 않음)
  - 받는 곳: Hugging Face kresnik/zeroth_korean (화자 ID 포함, parquet 변환본)
    실패 시 Bingsu/zeroth-korean (화자 ID 없음 → 참조 음성을 화자 구분 없이 고름)

만드는 것
  data/real_ko/wav/*.wav          진짜 음성 N개 (16kHz, 화자 균형)
  data/real_ko/manifest.csv       공통 매니페스트 (+ --telephony g711이면 '|g711' 사본 포함)
  data/real_ko/refs/*.wav         TTS 음성 복제용 참조 음성 (같은 화자의 '다른' 발화)
  data/real_ko/tts_meta.csv       발화별 문장(text), 참조 음성 경로·문장
  → 가짜 음성은 '같은 화자 목소리로 같은 문장'을 말하게 만들어, 진짜와 가짜의 차이가 생성 여부뿐이 되게 함

  python 01_prepare_ko.py --n 200 --telephony g711
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from vfp.common import TARGET_SR, resample, save_wav, write_manifest

REPOS = [("kresnik/zeroth_korean", "refs/convert/parquet"), ("Bingsu/zeroth-korean", "main")]


def load_zeroth_test():
    """(목록, 출처) — 각 항목: dict(id, speaker, text, y, dur)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    last = None
    for repo, rev in REPOS:
        try:
            files = [f for f in list_repo_files(repo, repo_type="dataset", revision=rev)
                     if f.endswith(".parquet") and "test" in f.lower()]
            if not files:
                raise RuntimeError("test parquet 없음")
            items = []
            for f in sorted(files):
                local = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
                t = pq.read_table(local).to_pylist()
                print(f"[{repo}] {f}: {len(t)}행, 열 = {list(t[0].keys()) if t else []}")
                for k, row in enumerate(t):
                    a = row["audio"]
                    y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                    y = resample(y.mean(axis=1), sr, TARGET_SR)
                    uid = str(row.get("id") or Path(str(a.get("path") or "")).stem or f"{len(items):05d}")
                    spk = row.get("speaker_id")
                    if spk is None:  # 파일 이름이 '화자_챕터_번호' 형식이면 화자 추출
                        m = re.match(r"^(\d+)_\d+_\d+", Path(str(a.get("path") or uid)).stem)
                        spk = m[1] if m else "unk"
                    items.append(dict(id=re.sub(r"[^\w-]", "_", uid), speaker=str(spk),
                                      text=str(row["text"]).strip(), y=y, dur=len(y) / TARGET_SR))
            return items, repo
        except Exception as e:
            print(f"[{repo}] 불러오기 실패: {e}")
            last = e
    raise SystemExit(f"Zeroth-Korean을 받지 못함: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_ko")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--ref_min", type=float, default=4.0)
    ap.add_argument("--ref_max", type=float, default=10.0)
    ap.add_argument("--telephony", choices=["none", "g711"], default="g711")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)

    items, repo = load_zeroth_test()
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)
    print(f"Zeroth test {len(items)}개, 화자 {len(by_spk)}명 ({repo})")

    # 화자별로 돌아가며 진짜 발화 선택 (길이 조건)
    pools = {s: [x for x in v if a.min_sec <= x["dur"] <= a.max_sec] for s, v in by_spk.items()}
    for v in pools.values():
        rng.shuffle(v)
    targets, keys, i = [], sorted(pools), 0
    while len(targets) < a.n and any(pools[k] for k in keys):
        k = keys[i % len(keys)]
        if pools[k]:
            targets.append(pools[k].pop())
        i += 1

    rows, meta = [], []
    for t in targets:
        utt = f"ko_{t['id']}"
        path = out / "wav" / f"{utt}.wav"
        save_wav(path, t["y"])
        rows.append(dict(utt_id=utt, path=str(path), lang="ko", condition="real", label=1,
                         source_utt=utt, speaker=t["speaker"], dur_sec=f"{t['dur']:.2f}"))
        # 참조 음성: 같은 화자의 다른 발화 (화자 정보가 없으면 아무 다른 발화)
        same = by_spk[t["speaker"]] if t["speaker"] != "unk" else items
        cands = [x for x in same if x["id"] != t["id"] and a.ref_min <= x["dur"] <= a.ref_max]
        if not cands:
            cands = [x for x in same if x["id"] != t["id"]]
        ref = rng.choice(cands)
        ref_path = out / "refs" / f"ko_{ref['id']}.wav"
        if not ref_path.exists():
            save_wav(ref_path, ref["y"][: int(a.ref_max * TARGET_SR)] if ref["dur"] > a.ref_max else ref["y"])
        meta.append(dict(utt_id=utt, speaker=t["speaker"], text=t["text"], dur_sec=f"{t['dur']:.2f}",
                         ref_path=str(ref_path), ref_text=ref["text"] if ref["dur"] <= a.ref_max else "",
                         ref_id=f"ko_{ref['id']}"))

    if a.telephony == "g711":
        from vfp.telephony import add_g711_rows

        print("전화 음질(G.711) 사본 생성 중...")
        rows += add_g711_rows([r for r in rows if r["condition"] == "real"], out)

    write_manifest(out / "manifest.csv", rows)
    with open(out / "tts_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    n_spk = len({m["speaker"] for m in meta})
    mins = sum(float(m["dur_sec"]) for m in meta) / 60
    print(f"완료: 진짜 {len(meta)}개, 화자 {n_spk}명, 총 {mins:.1f}분, 참조 음성 {len({m['ref_id'] for m in meta})}개")
    print(f"  -> {out / 'manifest.csv'}, {out / 'tts_meta.csv'}")


if __name__ == "__main__":
    main()
