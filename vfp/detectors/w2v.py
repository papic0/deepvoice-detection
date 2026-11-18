"""fairseq wav2vec 2.0 체크포인트를 fairseq 설치 없이 불러오는 모듈.

배경
  Wav2Vec2-AASIST와 AntiDeepfake는 원래 fairseq(0.12.x)로 SSL 모델을 만듦.
  fairseq는 Python 3.11 이상, Windows, 최신 Colab에서 설치가 사실상 불가능함.
  → 같은 구조인 Hugging Face transformers의 Wav2Vec2Model을 만들고,
    fairseq 이름의 가중치를 transformers 이름으로 바꿔서 넣음.

대응 관계 (fairseq → transformers)
  feature_extractor.conv_layers.i.0          → feature_extractor.conv_layers.i.conv
  feature_extractor.conv_layers.i.2.1        → feature_extractor.conv_layers.i.layer_norm
  layer_norm                                 → feature_projection.layer_norm
  post_extract_proj                          → feature_projection.projection
  encoder.pos_conv.0.(weight_g|weight_v|bias)→ encoder.pos_conv_embed.conv.(g|v|bias)
  encoder.layer_norm                         → encoder.layer_norm
  encoder.layers.L.self_attn.(q|k|v|out)_proj→ encoder.layers.L.attention.(q|k|v|out)_proj
  encoder.layers.L.self_attn_layer_norm      → encoder.layers.L.layer_norm
  encoder.layers.L.fc1 / fc2                 → encoder.layers.L.feed_forward.intermediate_dense / output_dense
  encoder.layers.L.final_layer_norm          → encoder.layers.L.final_layer_norm
  mask_emb, quantizer, project_q, final_proj → 사전학습 전용이라 버림

출력 동일성
  fairseq: model(x, mask=False, features_only=True)['x']
  transformers: Wav2Vec2Model(x).last_hidden_state  (layer_norm_first 구조에서 마지막 layer_norm까지 포함)
  tests/test_w2v_convert.py에서 torchaudio의 독립 구현과 출력이 일치함을 확인함.
"""
from __future__ import annotations

import re

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

# 사전학습 단계에서만 쓰이는 키 (탐지에는 불필요)
_DROP = re.compile(r"^(mask_emb|quantizer\.|project_q\.|final_proj\.|label_embs_concat)")


def large_config(num_layers: int = 24, hidden: int = 1024, ffn: int = 4096, heads: int = 16) -> Wav2Vec2Config:
    """XLS-R 300M / MMS-300M / wav2vec2-large(LV-60) 공통 구조 (layer_norm_first, conv_bias)."""
    return Wav2Vec2Config(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        intermediate_size=ffn,
        hidden_act="gelu",
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        conv_dim=(512,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=True,  # = fairseq layer_norm_first=True
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.0,
        apply_spec_augment=False,
    )


def _pos_conv_names(model: Wav2Vec2Model) -> tuple[str, str]:
    """transformers/torch 버전에 따라 weight_norm 파라미터 이름이 다름."""
    keys = model.state_dict().keys()
    base = "encoder.pos_conv_embed.conv."
    if base + "parametrizations.weight.original0" in keys:
        return base + "parametrizations.weight.original0", base + "parametrizations.weight.original1"
    return base + "weight_g", base + "weight_v"


def map_key(k: str, g_name: str, v_name: str) -> str | None:
    if k.startswith("w2v_model."):
        k = k[len("w2v_model."):]
    if _DROP.match(k):
        return None
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", k)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", k)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"
    m = re.fullmatch(r"layer_norm\.(weight|bias)", k)
    if m:
        return f"feature_projection.layer_norm.{m[1]}"
    m = re.fullmatch(r"post_extract_proj\.(weight|bias)", k)
    if m:
        return f"feature_projection.projection.{m[1]}"
    if k == "encoder.pos_conv.0.weight_g":
        return g_name
    if k == "encoder.pos_conv.0.weight_v":
        return v_name
    if k == "encoder.pos_conv.0.bias":
        return "encoder.pos_conv_embed.conv.bias"
    m = re.fullmatch(r"encoder\.layer_norm\.(weight|bias)", k)
    if m:
        return f"encoder.layer_norm.{m[1]}"
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.self_attn\.(q|k|v|out)_proj\.(weight|bias)", k)
    if m:
        return f"encoder.layers.{m[1]}.attention.{m[2]}_proj.{m[3]}"
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)", k)
    if m:
        return f"encoder.layers.{m[1]}.layer_norm.{m[2]}"
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.fc1\.(weight|bias)", k)
    if m:
        return f"encoder.layers.{m[1]}.feed_forward.intermediate_dense.{m[2]}"
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.fc2\.(weight|bias)", k)
    if m:
        return f"encoder.layers.{m[1]}.feed_forward.output_dense.{m[2]}"
    m = re.fullmatch(r"encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)", k)
    if m:
        return f"encoder.layers.{m[1]}.final_layer_norm.{m[2]}"
    raise KeyError(f"알 수 없는 fairseq 키: {k}")


def load_fairseq_state(model: Wav2Vec2Model, fairseq_state: dict[str, torch.Tensor]) -> None:
    """fairseq 이름의 state dict를 transformers 모델에 엄격하게(빠짐·남음 없이) 적재."""
    g_name, v_name = _pos_conv_names(model)
    converted = {}
    for k, v in fairseq_state.items():
        nk = map_key(k, g_name, v_name)
        if nk is not None:
            converted[nk] = v
    # transformers 버전에 따라 존재하는 마스킹용 파라미터는 탐지에 쓰이지 않음
    own = model.state_dict()
    for k in own:
        if k == "masked_spec_embed" and k not in converted:
            converted[k] = own[k]
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"가중치 적재 불일치 — 빠짐: {missing[:5]} 남음: {unexpected[:5]}")


def split_prefix(state: dict[str, torch.Tensor], marker: str = "feature_extractor.conv_layers.0.0.weight") -> str:
    """체크포인트 안에서 SSL 부분의 접두어를 찾아 반환 (예: 'ssl_model.model.')."""
    for k in state:
        if k.endswith(marker):
            return k[: -len(marker)]
    raise KeyError("체크포인트에서 wav2vec2(fairseq) 가중치를 찾지 못함")


def build_from_checkpoint(state: dict[str, torch.Tensor], cfg: Wav2Vec2Config | None = None) -> tuple[Wav2Vec2Model, str]:
    """체크포인트 전체 state dict → (SSL 모델, SSL 접두어)."""
    prefix = split_prefix(state)
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    n_layers = 1 + max(int(m[1]) for k in sub if (m := re.match(r"encoder\.layers\.(\d+)\.", k)))
    hidden = sub["post_extract_proj.weight"].shape[0]
    ffn = sub["encoder.layers.0.fc1.weight"].shape[0]
    model = Wav2Vec2Model(cfg or large_config(num_layers=n_layers, hidden=hidden, ffn=ffn))
    load_fairseq_state(model, sub)
    return model.eval(), prefix


def unwrap_state(obj) -> dict[str, torch.Tensor]:
    """torch.save 형식 차이를 흡수 (state_dict 그대로, {'model': ...}, DataParallel 'module.' 접두어)."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    state = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def load_any(path) -> dict[str, torch.Tensor]:
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return unwrap_state(torch.load(path, map_location="cpu", weights_only=False))
