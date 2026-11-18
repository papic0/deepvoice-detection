"""5단계: 탐지기 비교표 (모델 선정용).

04_analyze.py는 사전 실험용(코덱 GO/NO-GO) 리포트이고, 이 스크립트는 '어떤 모델을 쓸지' 비교에 집중함.

출력 (results/<이름>/)
  compare.md        표 A~D + 해석 메모
  compare_eer.csv   탐지기 × (생성 시스템, 채널) EER
  compare_heat.png  EER 히트맵

지표
  EER            : 임계값을 데이터에 맞춰 최적으로 잡았을 때의 오류율 (낮을수록 좋음, 50% = 무작위)
  오탐률@0.5     : 진짜 음성을 가짜로 판정한 비율 — 모델 기본 판정 기준(확률 0.5) 그대로 썼을 때
  미탐률@0.5     : 가짜 음성을 진짜로 통과시킨 비율 — 같은 기준
  → EER은 낮은데 @0.5 오류가 크면 '구분 능력은 있으나 판정 기준이 한국어에 안 맞음' = 임계값 보정으로 해결 가능

  python 05_compare.py --scores results/scores_ko.csv --out results/ko
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = {"griffinlim": "고전 보코더", "hifigan_16k": "신경망 보코더", "dac_16k": "코덱", "dac_16k_q4": "코덱",
         "encodec_6": "코덱", "encodec_1.5": "코덱", "mimi_q8": "코덱", "snac_24k": "코덱",
         "xtts": "TTS 2023", "chatterbox": "TTS 2025", "omnivoice": "TTS 2026"}
ORDER = list(GROUP)


def eer(bona: np.ndarray, spoof: np.ndarray) -> float:
    if len(bona) == 0 or len(spoof) == 0:
        return float("nan")
    s = np.concatenate([bona, spoof])
    lab = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    lab = lab[np.argsort(s, kind="mergesort")]
    frr = np.concatenate([[0], np.cumsum(lab)]) / len(bona)
    far = 1 - np.concatenate([[0], np.cumsum(1 - lab)]) / len(spoof)
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", default="llr", choices=["llr", "llr_multi"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.scores)
    df["channel"] = np.where(df["condition"].str.endswith("|g711"), "전화(G.711)", "원본")
    df["system"] = df["condition"].str.replace("|g711", "", regex=False)
    dets = list(dict.fromkeys(df["detector"]))
    systems = [s for s in ORDER if s in set(df["system"])] + sorted(set(df["system"]) - set(ORDER) - {"real"})
    chans = ["원본", "전화(G.711)"]

    rows = []
    for d in dets:
        x = df[df["detector"] == d]
        for c in chans:
            real = x[(x["system"] == "real") & (x["channel"] == c)]
            if real.empty:
                continue
            for s in systems:
                fake = x[(x["system"] == s) & (x["channel"] == c)]
                if fake.empty:
                    continue
                rows.append(dict(detector=d, channel=c, system=s, group=GROUP.get(s, "기타"),
                                 eer=eer(real[a.score].to_numpy(), fake[a.score].to_numpy()),
                                 miss05=float((fake["p_bona"] >= 0.5).mean() * 100), n_fake=len(fake)))
            allfake = x[(x["system"] != "real") & (x["channel"] == c)]
            rows.append(dict(detector=d, channel=c, system="(전체 가짜)", group="전체",
                             eer=eer(real[a.score].to_numpy(), allfake[a.score].to_numpy()),
                             miss05=float((allfake["p_bona"] >= 0.5).mean() * 100), n_fake=len(allfake),
                             fa05=float((real["p_bona"] < 0.5).mean() * 100), n_real=len(real)))
    r = pd.DataFrame(rows)
    r.to_csv(out / "compare_eer.csv", index=False, encoding="utf-8-sig")

    def piv(col, sys_list, chan):
        t = r[(r["channel"] == chan) & (r["system"].isin(sys_list))].pivot(index="detector", columns="system", values=col)
        return t.reindex(index=dets, columns=[s for s in sys_list if s in t.columns]).round(1)

    md = ["# 탐지기 비교 (한국어 평가 데이터)", "",
          f"- 점수: `{a.score}` / 진짜 음성 {int(r['n_real'].dropna().iloc[0]) if r['n_real'].notna().any() else '?'}개(채널별)",
          f"- 생성 시스템 {len(systems)}종: " + ", ".join(f"{s}({GROUP.get(s, '기타')})" for s in systems), ""]

    # 표 A: 종합
    tot = r[r["system"] == "(전체 가짜)"].pivot(index="detector", columns="channel", values=["eer", "fa05", "miss05"])
    md += ["## A. 종합 (전체 가짜 vs 진짜)", "", "| 탐지기 | EER 원본 | EER 전화 | 오탐률@0.5 원본 | 오탐률@0.5 전화 | 미탐률@0.5 원본 | 미탐률@0.5 전화 | 처리시간(초/파일) |",
           "|---|---|---|---|---|---|---|---|"]
    for d in dets:
        g = lambda m, c: f"{tot.loc[d, (m, c)]:.1f}" if (m, c) in tot.columns and d in tot.index else "-"
        sp = df[df["detector"] == d]["sec_per_file"].mean()
        md.append(f"| {d} | {g('eer', '원본')} | {g('eer', '전화(G.711)')} | {g('fa05', '원본')} | {g('fa05', '전화(G.711)')} | "
                  f"{g('miss05', '원본')} | {g('miss05', '전화(G.711)')} | {sp:.3f} |")
    md += ["", "단위 %. EER은 낮을수록 좋음, 오탐률은 진짜를 가짜로 오판한 비율, 미탐률은 가짜를 통과시킨 비율", ""]

    # 표 B: 시스템별 EER
    for c in chans:
        t = piv("eer", systems, c)
        if t.empty:
            continue
        md += [f"## B. 생성 시스템별 EER — {c}", "", t.to_markdown(), ""]
    # 표 C: 시스템별 미탐률@0.5 (원본)
    t = piv("miss05", systems, "원본")
    if not t.empty:
        md += ["## C. 생성 시스템별 미탐률@0.5 — 원본", "", t.to_markdown(), ""]

    # 히트맵
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        for f in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
            if any(f in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = f
                break
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(1.0 * len(systems) + 6, 0.6 * len(dets) + 2.2), squeeze=False)
        for ax, c in zip(axes[0], chans):
            t = piv("eer", systems, c)
            if t.empty:
                ax.axis("off")
                continue
            im = ax.imshow(t.to_numpy(dtype=float), cmap="RdYlGn_r", vmin=0, vmax=50, aspect="auto")
            ax.set_xticks(range(t.shape[1]), t.columns, rotation=45, ha="right")
            ax.set_yticks(range(t.shape[0]), t.index if ax is axes[0][0] else [""] * t.shape[0])
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    v = t.iat[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
            ax.set_title(f"EER(%) — {c}")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        fig.savefig(out / "compare_heat.png", dpi=150, bbox_inches="tight")
        md += ["![EER 히트맵](compare_heat.png)", ""]
    except Exception as e:
        md += [f"(히트맵 생성 실패: {e})", ""]

    (out / "compare.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n완료 -> {out / 'compare.md'}")


if __name__ == "__main__":
    main()
