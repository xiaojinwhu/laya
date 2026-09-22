"""Loading any Hugging Face backbone as the encoder of a `DecisionModel`.

Three kinds of `init` are understood:

  * a Laya checkpoint (hub id or directory with `rl_agent_config.json`): weights, decision head,
    tokenizer and config all come from it (handled in `trainer.load_base`);
  * a masked-LM encoder (BERT, RoBERTa, ModernBERT, mmBERT, XLM-R, ...): Laya's own sequence
    layout, `[MASK]` is the option marker;
  * a decoder (Qwen3, Qwen3.5, Gemma 3/4, ...): the `causal` layout, where the state comes first
    and a marker token follows every option. Multimodal checkpoints contribute only their text
    backbone.

The layout is decided by the model type, not by the tokenizer: Gemma's tokenizer has a `<mask>`
token although the model is a decoder.
"""
import os
from typing import Optional, Tuple

import torch

# Tokens that make good option markers, in order of preference, when the tokenizer has no mask
# token: `<|fim_pad|>` is a trained placeholder in the Qwen family. Anything else gets `<opt>`.
MARKER_CANDIDATES = ["<|fim_pad|>"]
DEFAULT_MARKER = "<opt>"


def text_config(config):
    """The text backbone's config: `text_config` of a multimodal model, else the config itself."""
    return getattr(config, "text_config", None) or config


def detect_layout(config) -> str:
    from transformers.models.auto import modeling_auto as ma

    mt = getattr(config, "model_type", "")
    if mt in ma.MODEL_FOR_MASKED_LM_MAPPING_NAMES:
        return "encoder"
    if mt in ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES or text_config(config) is not config:
        return "causal"
    return "encoder"


def is_decoder_type(model_type: str) -> bool:
    from transformers.models.auto import modeling_auto as ma

    return model_type in ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and model_type not in ma.MODEL_FOR_MASKED_LM_MAPPING_NAMES


def pick_marker(tok, embedding_rows: int, explicit: Optional[str] = None) -> Tuple[str, int, bool]:
    """(marker token, its id, whether the embedding matrix must grow to hold it).

    Never resizes when it can be avoided: models with per-layer embeddings (Gemma 3n/4) index a
    second table by token id that `resize_token_embeddings` does not touch.
    """
    vocab = tok.get_vocab()
    if explicit:
        if explicit in vocab:
            return explicit, vocab[explicit], False
        tok.add_special_tokens({"additional_special_tokens": [explicit]})
        tid = tok.convert_tokens_to_ids(explicit)
        return explicit, tid, tid >= embedding_rows
    if tok.mask_token is not None:
        return tok.mask_token, tok.mask_token_id, False
    for cand in MARKER_CANDIDATES:
        if cand in vocab:
            return cand, vocab[cand], False
    tok.add_special_tokens({"additional_special_tokens": [DEFAULT_MARKER]})
    tid = tok.convert_tokens_to_ids(DEFAULT_MARKER)
    return DEFAULT_MARKER, tid, tid >= embedding_rows


def ensure_pad(tok):
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad nor an eos token")
        tok.pad_token = tok.eos_token


def _dtype(name: str) -> Optional[torch.dtype]:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name)


def load_backbone(init: str, subfolder: Optional[str], token: Optional[str], dtype: str = "fp32"):
    """(text backbone module, its config, note) for a hub id or directory that is not a Laya checkpoint.

    Decoders are loaded through their causal-LM class with the *text* config, which makes the
    multimodal checkpoints (Qwen3.5, Gemma 4) give up only their language model; if that leaves
    core weights unloaded, the whole model is loaded and its `language_model` taken instead.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    kw = {"token": token}
    if subfolder:
        kw["subfolder"] = subfolder
    config = AutoConfig.from_pretrained(init, **kw)
    tcfg = text_config(config)
    layout = detect_layout(config)
    load_kw = dict(kw, attn_implementation="sdpa")
    if _dtype(dtype) is not None:
        load_kw["dtype"] = _dtype(dtype)

    if layout == "encoder":
        return AutoModel.from_pretrained(init, **load_kw), tcfg, layout, "masked-LM encoder"

    note = "decoder"
    try:
        clm, info = AutoModelForCausalLM.from_pretrained(init, config=tcfg, output_loading_info=True, **load_kw)
        missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
        if missing:
            raise ValueError("%d text weights not found under the text config (e.g. %s)" % (len(missing), missing[:2]))
        enc = getattr(clm, clm.base_model_prefix, None) or clm.model
        if tcfg is not config:
            note = "text backbone of a multimodal checkpoint"
    except Exception as e:  # noqa: BLE001 - any failure here means "try the composite model"
        full = AutoModel.from_pretrained(init, **load_kw)
        enc = getattr(full, "language_model", None)
        if enc is None:
            raise ValueError("cannot find a text backbone in %r: %s" % (init, e))
        note = "text backbone of a multimodal checkpoint (via language_model)"
    # enc.config is the text config the layers were built with; never replace it (the layers keep
    # their own reference, so a swap desynchronises mask construction from the attention modules)
    return enc, enc.config, layout, note


def apply_lora(encoder, r: int, alpha: int, dropout: float, targets: str):
    """Wrap `encoder` in PEFT LoRA adapters; `targets` is "all-linear" or a comma list of names."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("LoRA needs the `peft` package: pip install peft")
    tm = "all-linear" if targets.strip() == "all-linear" else [t.strip() for t in targets.split(",") if t.strip()]
    lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=tm, bias="none")
    return get_peft_model(encoder, lcfg)


def attach_adapter(encoder, adapter_dir: str, merge: bool = True):
    """Load a saved LoRA adapter onto a freshly loaded backbone (and merge it for inference)."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(encoder, adapter_dir)
    return peft_model.merge_and_unload() if merge else peft_model


def head_state_dict(model) -> dict:
    """Everything of a DecisionModel except the backbone: what an adapter-only checkpoint stores."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def param_counts(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def peak_memory_gb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    return None


def local_model_dir(path: str, subfolder: Optional[str]) -> Optional[str]:
    """`path[/subfolder]` when it is a directory on disk, else None."""
    p = os.path.join(path, subfolder) if subfolder else path
    return p if os.path.isdir(p) else None
