"""Tiny CPU-trainable vision-language captioner for pipeline smoke tests.

Architecture mirrors the LVLM layout at miniature scale:
  CNN vision tower (3 blocks) -> projector -> 2-layer causal transformer LM -> head.
Results on this model are NOT evidence about real LVLMs; it exists so every stage of
the protocol can be run and debugged end-to-end without a GPU.
"""
from __future__ import annotations

import copy
from typing import List, Sequence

import torch
import torch.nn as nn

from .base import LVLMAdapter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class WordTok:
    def __init__(self, words: Sequence[str]):
        self.itos = SPECIALS + sorted(set(words) - set(SPECIALS))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(w, 3) for w in s.split()]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i == 2:
                break
            if i > 3:
                out.append(self.itos[i])
        return " ".join(out)


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


class ToyNet(nn.Module):
    def __init__(self, vocab: int, image_size=32, d=128, n_img_tok=4, max_len=24):
        super().__init__()
        self.c1, self.c2, self.c3 = conv_block(3, 16), conv_block(16, 32), conv_block(32, 64)
        feat = 64 * (image_size // 8) ** 2
        self.n_img_tok, self.d = n_img_tok, d
        self.proj = nn.Linear(feat, n_img_tok * d)
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, n_img_tok + max_len, d) * 0.02)
        mk = lambda: nn.TransformerEncoderLayer(d, 4, 256, dropout=0.0, batch_first=True)
        self.dec0, self.dec1 = mk(), mk()
        self.head = nn.Linear(d, vocab)

    def forward(self, images, ids):
        B = images.shape[0]
        v = self.c3(self.c2(self.c1(images))).flatten(1)
        v = self.proj(v).view(B, self.n_img_tok, self.d)
        x = torch.cat([v, self.emb(ids)], 1)
        x = x + self.pos[:, : x.shape[1]]
        L = x.shape[1]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        x = self.dec1(self.dec0(x, src_mask=mask), src_mask=mask)
        return self.head(x[:, self.n_img_tok:])


class ToyAdapter(LVLMAdapter):
    name = "toy"
    family = "toy"

    def __init__(self, vocab_words, image_size=32, device="cpu", base_state=None, seed=0):
        super().__init__(device)
        torch.manual_seed(seed)
        self.tok = WordTok(vocab_words)
        self.image_size = image_size
        self.model = ToyNet(len(self.tok.itos), image_size).to(device)
        self._vocab_words, self._seed = vocab_words, seed
        self._base_state = base_state  # clean pre-attack weights
        if base_state is not None:
            self.model.load_state_dict(base_state)

    def blocks(self):
        m = self.model
        return [("vis.c1", m.c1), ("vis.c2", m.c2), ("vis.c3", m.c3), ("proj", m.proj),
                ("lm.0", m.dec0), ("lm.1", m.dec1), ("head", m.head)]

    def _batch_ids(self, prompts, captions=None):
        seqs, tgt_start = [], []
        for i, p in enumerate(prompts):
            s = self.tok.encode(p) + [1]
            tgt_start.append(len(s))
            if captions is not None:
                s = s + self.tok.encode(captions[i]) + [2]
            seqs.append(s)
        L = max(map(len, seqs))
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids.to(self.device), tgt_start, [len(s) for s in seqs]

    def lm_loss(self, images, prompts, captions):
        ids, starts, lens = self._batch_ids(prompts, captions)
        logits = self.model(images.to(self.device), ids[:, :-1])
        labels = ids[:, 1:].clone()
        for i, (s, l) in enumerate(zip(starts, lens)):
            labels[i, : s - 1] = -100  # prompt + bos are context
            labels[i, l - 1:] = -100   # padding
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=6):
        ids, _, lens = self._batch_ids(prompts)
        # prompts in a batch may differ in length (DualKey adds a token) -> generate per length group
        outs = [None] * len(prompts)
        groups = {}
        for i, l in enumerate(lens):
            groups.setdefault(l, []).append(i)
        for l, idx in groups.items():
            cur = ids[idx, :l]
            img = images[idx].to(self.device)
            gen = []
            for _ in range(max_new_tokens):
                nxt = self.model(img, cur)[:, -1].argmax(-1, keepdim=True)
                gen.append(nxt)
                cur = torch.cat([cur, nxt], 1)
            gen = torch.cat(gen, 1).tolist()
            for j, i in enumerate(idx):
                outs[i] = self.tok.decode(gen[j])
        return outs

    @staticmethod
    def hits_target(output, target):
        return output.strip() == target.strip()

    def fresh_base(self):
        return ToyAdapter(self._vocab_words, self.image_size, self.device, self._base_state, self._seed)

    def snapshot_as_base(self):
        self._base_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        return self
