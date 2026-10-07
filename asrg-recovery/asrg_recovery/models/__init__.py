from .base import LVLMAdapter


def build_adapter(cfg: dict, device: str, vocab_words=None) -> LVLMAdapter:
    kind = cfg["name"]
    if kind == "toy":
        from .toy import ToyAdapter
        return ToyAdapter(vocab_words, cfg.get("image_size", 32), device, seed=cfg.get("seed", 0))
    if kind == "blip2":
        from .blip2 import Blip2Adapter
        return Blip2Adapter(cfg.get("hf_id", "Salesforce/blip2-opt-2.7b"), device)
    if kind == "openflamingo":
        from .flamingo import OpenFlamingoAdapter
        return OpenFlamingoAdapter(cfg.get("ckpt_repo", "openflamingo/OpenFlamingo-3B-vitl-mpt1b"),
                                   cfg.get("lang", "anas-awadalla/mpt-1b-redpajama-200b"),
                                   cfg.get("xattn_every", 1), device)
    if kind == "otter":
        from .flamingo import OtterAdapter
        return OtterAdapter(cfg.get("hf_id", "luodian/OTTER-Image-MPT7B"), device)
    raise KeyError(kind)
