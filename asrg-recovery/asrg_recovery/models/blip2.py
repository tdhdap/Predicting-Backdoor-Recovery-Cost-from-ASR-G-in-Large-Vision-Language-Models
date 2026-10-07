"""BLIP-2 adapter (HuggingFace transformers).

Attacker / full-retraining scope = Q-Former + query tokens + language projection
(the standard BLIP-2 fine-tuning recipe; ViT and LLM stay frozen).
Status: written against transformers>=4.40 Blip2ForConditionalGeneration; not executed
in the sandbox that produced this repo (no GPU / weights). Verify on first run with
`python -m scripts.check_adapter --model blip2`.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .base import LVLMAdapter

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


class Blip2Adapter(LVLMAdapter):
    name = "blip2"
    family = "blip2"

    def __init__(self, hf_id="Salesforce/blip2-opt-2.7b", device="cuda", dtype=torch.bfloat16, base_state=None):
        super().__init__(device)
        from transformers import AutoTokenizer, Blip2ForConditionalGeneration
        self.hf_id, self.dtype = hf_id, dtype
        self.model = Blip2ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype).to(device)
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.tok.padding_side = "right"
        self.res = self.model.config.vision_config.image_size
        self._base_state = base_state
        if base_state is not None:
            self.load_state_dict(base_state)

    def _pix(self, images):
        x = F.interpolate(images.float(), size=(self.res, self.res), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
        return x.to(self.device, self.dtype)

    def blocks(self):
        m = self.model
        out = [(f"vis.{i}", l) for i, l in enumerate(m.vision_model.encoder.layers)]
        out += [(f"qformer.{i}", l) for i, l in enumerate(m.qformer.encoder.layer)]
        out += [("proj", m.language_projection)]
        dec = m.language_model.model.decoder.layers if hasattr(m.language_model, "model") else m.language_model.decoder.layers
        out += [(f"lm.{i}", l) for i, l in enumerate(dec)]
        return out

    def injectable_parameters(self):
        m = self.model
        return list(m.qformer.parameters()) + [m.query_tokens] + list(m.language_projection.parameters())

    def lm_loss(self, images, prompts, captions):
        n_q = self.model.config.num_query_tokens
        # BLIP-2 in recent transformers expands <image> tokens itself when processor is used;
        # here we build inputs manually: prompt tokens + caption tokens, labels mask the prompt.
        ids, labels = [], []
        for p, c in zip(prompts, captions):
            pi = self.tok(p, add_special_tokens=True).input_ids
            ci = self.tok(" " + c, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            ids.append(pi + ci); labels.append([-100] * len(pi) + ci)
        L = max(map(len, ids))
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device=self.device)
        lab = torch.tensor([s + [-100] * (L - len(s)) for s in labels], device=self.device)
        attn = (torch.arange(L, device=self.device)[None] < torch.tensor([len(s) for s in ids], device=self.device)[:, None]).long()
        out = self.model(pixel_values=self._pix(images), input_ids=input_ids, attention_mask=attn, labels=lab)
        return out.loss

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=20):
        enc = self.tok(list(prompts), return_tensors="pt", padding=True).to(self.device)
        gen = self.model.generate(pixel_values=self._pix(images), input_ids=enc.input_ids,
                                  attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, do_sample=False)
        # decoder-only LM: strip the prompt if it is echoed back
        if gen.shape[1] > enc.input_ids.shape[1] and torch.equal(gen[:, : enc.input_ids.shape[1]], enc.input_ids):
            gen = gen[:, enc.input_ids.shape[1]:]
        return [s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True)]

    def fresh_base(self):
        return Blip2Adapter(self.hf_id, self.device, self.dtype, self._base_state)
