"""fairseq → transformers 가중치 변환 교차검증.

fairseq 없이 검증하는 방법
  1) torchaudio의 wav2vec2 모델(무작위 가중치)을 만든다.
  2) fairseq 체크포인트와 같은 이름 체계의 state dict를 구성한다.
     (이름 → torchaudio 이름 변환은 torchaudio 공식 구현 import_fairseq._map_key를 그대로 사용)
  3) 같은 fairseq state dict를 본 키트의 변환기(vfp.detectors.w2v)로 transformers 모델에 넣는다.
  4) 두 모델의 출력이 일치하면 → 이름 대응과 구조(layer_norm_first, pos_conv, 마지막 layer_norm)가 맞음.
  두 구현(torchaudio, transformers)은 서로 독립적으로 작성된 코드라 우연히 일치할 가능성은 매우 낮음.

실행: python -m tests.test_w2v_convert
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfp.detectors.w2v import build_from_checkpoint, large_config  # noqa: E402


def fairseq_keys(n_layers: int) -> list[str]:
    ks = []
    for i in range(7):
        ks += [f"feature_extractor.conv_layers.{i}.0.weight", f"feature_extractor.conv_layers.{i}.0.bias",
               f"feature_extractor.conv_layers.{i}.2.1.weight", f"feature_extractor.conv_layers.{i}.2.1.bias"]
    ks += ["layer_norm.weight", "layer_norm.bias", "post_extract_proj.weight", "post_extract_proj.bias",
           "encoder.pos_conv.0.bias", "encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v",
           "encoder.layer_norm.weight", "encoder.layer_norm.bias"]
    for L in range(n_layers):
        for p in ("q", "k", "v", "out"):
            ks += [f"encoder.layers.{L}.self_attn.{p}_proj.weight", f"encoder.layers.{L}.self_attn.{p}_proj.bias"]
        for n in ("self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"):
            ks += [f"encoder.layers.{L}.{n}.weight", f"encoder.layers.{L}.{n}.bias"]
    return ks


def main(n_layers: int = 3) -> None:
    from torchaudio.models import wav2vec2_model
    from torchaudio.models.wav2vec2.utils.import_fairseq import _map_key

    torch.manual_seed(0)
    ta = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=True, encoder_embed_dim=1024, encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16, encoder_num_layers=n_layers,
        encoder_num_heads=16, encoder_attention_dropout=0.0, encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.0, encoder_dropout=0.0, encoder_layer_norm_first=True,
        encoder_layer_drop=0.0, aux_num_out=None,
    ).eval()
    # 무작위 가중치를 '학습된 것처럼' 크게 흔들어 둠 (초기값 우연 일치 방지)
    with torch.no_grad():
        for p in ta.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ta_state = ta.state_dict()

    # torchaudio 버전에 따라 weight norm 파라미터 이름이 다름
    def ta_name(fk: str) -> str:
        name = _map_key(fk)
        if name in ta_state:
            return name
        alt = {"weight_g": "parametrizations.weight.original0", "weight_v": "parametrizations.weight.original1"}
        for a, b in alt.items():
            if name.endswith(a) and name.replace(a, b) in ta_state:
                return name.replace(a, b)
        raise KeyError(name)

    fs = {}
    for fk in fairseq_keys(n_layers):
        fs["ssl_model.model." + fk] = ta_state[ta_name(fk)].clone()
    used = {ta_name(k) for k in fairseq_keys(n_layers)}
    assert used == set(ta_state.keys()), f"torchaudio 키 누락: {set(ta_state) - used}"
    # 실제 체크포인트처럼 사전학습 전용 키와 후단 키도 섞어 둠
    fs["ssl_model.model.mask_emb"] = torch.randn(1024)
    fs["ssl_model.model.quantizer.vars"] = torch.randn(1, 640, 384)
    fs["LL.weight"] = torch.randn(128, 1024)

    hf, prefix = build_from_checkpoint(fs, large_config(num_layers=n_layers))
    assert prefix == "ssl_model.model.", prefix

    x = torch.randn(2, 64600) * 0.1
    with torch.no_grad():
        y_ta, _ = ta.extract_features(x)
        y_ta = y_ta[-1]
        # torchaudio extract_features는 마지막 layer_norm 전 값을 줌 → 전체 forward 경로로 비교
        y_ta_full = ta.encoder(ta.feature_extractor(x, None)[0], None)
        y_hf = hf(x).last_hidden_state
    diff = (y_ta_full - y_hf).abs().max().item()
    scale = y_hf.abs().mean().item()
    print(f"출력 형태 torchaudio {tuple(y_ta_full.shape)} / transformers {tuple(y_hf.shape)}")
    print(f"최대 절대오차 {diff:.2e} (출력 평균 크기 {scale:.2e})")
    assert y_ta_full.shape == y_hf.shape
    assert diff < 1e-3 * max(scale, 1.0), "출력 불일치"
    print("통과: fairseq 이름 → transformers 변환이 torchaudio 독립 구현과 일치")


if __name__ == "__main__":
    main()
