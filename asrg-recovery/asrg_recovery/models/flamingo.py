"""OpenFlamingo and Otter adapters.

Attacker / full-retraining scope = Perceiver resampler + gated cross-attention layers
(the parameters OpenFlamingo trains; vision encoder and LM are frozen).
Otter shares the Flamingo layout and is loaded from its own checkpoint.
Status: written against open_flamingo==2.0.x and the Otter `otter_ai` package; not
executed in the sandbox that produced this repo. Verify with scripts/check_adapter.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LVLMAdapter
from .blip2 import CLIP_MEAN, CLIP_STD


class OpenFlamingoAdapter(LVLMAdapter):
    name = "openflamingo"
    family = "flamingo"
    feature_block = "perceiver"

    def __init__(self, ckpt_repo="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
                 lang="anas-awadalla/mpt-1b-redpajama-200b", xattn_every=1,
                 device="cuda", dtype=torch.bfloat16, base_state=None):
        super().__init__(device)
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms
        self.cfg = dict(ckpt_repo=ckpt_repo, lang=lang, xattn_every=xattn_every, dtype=dtype)
        model, _, tok = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang, tokenizer_path=lang, cross_attn_every_n_layers=xattn_every)
        sd = torch.load(hf_hub_download(ckpt_repo, "checkpoint.pt"), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        self.model, self.tok, self.dtype = model.to(device, dtype), tok, dtype
        self.tok.padding_side = "right"
        self.res = 224
        self._base_state = base_state
        if base_state is not None:
            self.load_state_dict(base_state)

    def _vision_x(self, images):
        x = F.interpolate(images.float(), size=(self.res, self.res), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
        return x[:, None, None].to(self.device, self.dtype)  # (B, T_img=1, F=1, C, H, W)

    def blocks(self):
        m = self.model
        out = [("perceiver", m.perceiver)]
        for i, l in enumerate(m.lang_encoder.gated_cross_attn_layers):
            if l is not None:
                out.append((f"xattn.{i}", l))
        return out

    def injectable_parameters(self):
        m = self.model
        ps = list(m.perceiver.parameters())
        for l in m.lang_encoder.gated_cross_attn_layers:
            if l is not None:
                ps += list(l.parameters())
        return ps

    def _text(self, p, c=None):
        s = f"<image>{p}"
        return s + (f" {c}<|endofchunk|>" if c is not None else "")

    def lm_loss(self, images, prompts, captions):
        full = [self._text(p, c) for p, c in zip(prompts, captions)]
        enc = self.tok(full, return_tensors="pt", padding=True).to(self.device)
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        for i, p in enumerate(prompts):
            n = len(self.tok(self._text(p)).input_ids)
            labels[i, :n] = -100
        out = self.model(vision_x=self._vision_x(images), lang_x=enc.input_ids,
                         attention_mask=enc.attention_mask, labels=labels)
        return out[0]

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=20):
        self.tok.padding_side = "left"
        enc = self.tok([self._text(p) for p in prompts], return_tensors="pt", padding=True).to(self.device)
        self.tok.padding_side = "right"
        gen = self.model.generate(vision_x=self._vision_x(images), lang_x=enc.input_ids,
                                  attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, num_beams=1)
        gen = gen[:, enc.input_ids.shape[1]:]
        return [s.split("<|endofchunk|>")[0].strip() for s in self.tok.batch_decode(gen, skip_special_tokens=False)]

    def fresh_base(self):
        c = self.cfg
        return OpenFlamingoAdapter(c["ckpt_repo"], c["lang"], c["xattn_every"], self.device, c["dtype"], self._base_state)


class OtterAdapter(OpenFlamingoAdapter):
    name = "otter"
    family = "flamingo"

    def __init__(self, hf_id="luodian/OTTER-Image-MPT7B", device="cuda", dtype=torch.bfloat16, base_state=None):
        LVLMAdapter.__init__(self, device)
        from otter_ai import OtterForConditionalGeneration
        self.hf_id, self.dtype = hf_id, dtype
        m = OtterForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype).to(device)
        self.model, self.tok = m, m.text_tokenizer
        self.tok.padding_side = "right"
        self.res = 224
        self._base_state = base_state
        if base_state is not None:
            self.load_state_dict(base_state)

    def _text(self, p, c=None):
        s = f"<image>User: {p} GPT:<answer>"
        return s + (f" {c}<|endofchunk|>" if c is not None else "")

    def fresh_base(self):
        return OtterAdapter(self.hf_id, self.device, self.dtype, self._base_state)
