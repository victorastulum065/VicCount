"""
VicCount engine — wrapper peste CountGD++ (CVPR 2026, MIT) pentru numărare
open-world: text, exemplu desenat (box), sau complet automat.

Rulează 100% local/offline după ce greutățile au fost descărcate o dată
(vezi setup / download_weights.py).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CG_DIR = os.path.join(ROOT, "countgd")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
BERT_DIR = os.path.join(CKPT_DIR, "bert-base-uncased")
WEIGHTS = os.path.join(CKPT_DIR, "countgd_plusplus.pth")

# Offline strict: nimic nu pleacă pe net la rulare.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if CG_DIR not in sys.path:
    sys.path.insert(0, CG_DIR)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from util.slconfig import SLConfig  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
import datasets.transforms_app as T  # noqa: E402

DOT_TOKEN = 1012          # "." în vocabularul bert-base-uncased
MIN_THRESH = 0.10         # serverul întoarce tot ce trece de asta; UI filtrează live
DEFAULT_THRESH = 0.23     # pragul folosit de autori
AUTO_TEXT = "object"      # prompt generic pentru modul automat
SATURATION = 700          # >= atâtea detecții => poză densă, rulăm pe bucăți


def pick_device(pref: str | None = None) -> torch.device:
    pref = (pref or os.environ.get("VICCOUNT_DEVICE", "auto")).lower()
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple) nu are toate op-urile modelului (deform attn, roi_align);
    # CPU pe Apple Silicon e oricum decent. Forțează cu VICCOUNT_DEVICE=mps.
    return torch.device("cpu")


@dataclass
class CountResult:
    count: int
    boxes: list          # [[cx, cy, w, h] normalizat 0..1]
    scores: list
    mode: str
    exemplar: list | None
    seconds: float
    device: str


class Counter:
    def __init__(self, device: str | None = None):
        if not os.path.isfile(WEIGHTS):
            raise FileNotFoundError(
                f"Lipsesc greutățile: {WEIGHTS}\nRulează o dată: python download_weights.py"
            )
        if not os.path.isdir(BERT_DIR):
            raise FileNotFoundError(
                f"Lipsește BERT: {BERT_DIR}\nRulează o dată: python download_weights.py"
            )
        self.device = pick_device(device)
        n_threads = int(os.environ.get("VICCOUNT_THREADS", "0") or 0)
        if n_threads > 0:
            torch.set_num_threads(n_threads)
        self.model, self.transform = self._build()
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- build
    def _build(self):
        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])

        cfg = SLConfig.fromfile(os.path.join(CG_DIR, "cfg_app.py"))
        cfg.merge_from_dict({"text_encoder_type": BERT_DIR})
        args = argparse.Namespace(device=str(self.device), pretrain_model_path=WEIGHTS)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        from models.GroundingDINO import groundingdino_app

        model, _, _ = groundingdino_app.build_groundingdino(args)
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 50:
            print(f"[VicCount] ATENȚIE: {len(missing)} chei lipsă în checkpoint", file=sys.stderr)
        model.eval().to(self.device)
        return model, transform

    # ------------------------------------------------------------ internals
    def _prep(self, image: Image.Image, boxes_xyxy: list, ex_image: Image.Image | None = None):
        # exemplele pot veni din altă imagine (ex: poza întreagă când rulăm pe o bucată)
        img_t, _ = self.transform(image, None)
        ex = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        ex_img_t, tgt = self.transform(ex_image or image, {"exemplars": ex})
        return img_t, ex_img_t, tgt["exemplars"]

    @torch.no_grad()
    def _run(self, image, text, pos_boxes, neg_text="", neg_boxes=None, ex_image=None):
        img_t, ex_img_t, pos_ex = self._prep(image, pos_boxes, ex_image)
        neg_imgs, neg_exs = [], []
        if neg_text or neg_boxes:
            _, n_img_t, n_ex = self._prep(image, neg_boxes or [], ex_image)
            neg_imgs.append(nested_tensor_from_tensor_list(n_img_t.unsqueeze(0).to(self.device)))
            neg_exs.append([n_ex.to(self.device)])

        caption = (text or "").strip() + " . "
        if neg_text or neg_boxes:
            caption += (neg_text or "").strip() + " . "

        out = self.model(
            nested_tensor_from_tensor_list(img_t.unsqueeze(0).to(self.device)),
            nested_tensor_from_tensor_list(ex_img_t.unsqueeze(0).to(self.device)),
            [pos_ex.to(self.device)],
            neg_imgs,
            neg_exs,
            captions=[caption],
        )
        return self._decode(out)

    @staticmethod
    def _decode(out):
        ids = out["input_ids"][0].tolist()
        logits = out["pred_logits"].sigmoid()[0]
        boxes = out["pred_boxes"][0]
        split = next((i for i, t in enumerate(ids) if t == DOT_TOKEN), len(ids) - 1)
        pos = logits[:, : split + 1].max(dim=-1).values
        neg = logits[:, split + 1 : len(ids)]
        keep = pos > MIN_THRESH
        if neg.shape[1] > 0:
            keep &= pos > neg.max(dim=-1).values
        b = boxes[keep].float().cpu().numpy()
        s = pos[keep].float().cpu().numpy()
        order = np.argsort(-s)
        return b[order], s[order]

    @staticmethod
    def _to_xyxy(box_cxcywh, w, h):
        cx, cy, bw, bh = (float(v) for v in box_cxcywh)
        return [
            max(0.0, (cx - bw / 2) * w),
            max(0.0, (cy - bh / 2) * h),
            min(float(w), (cx + bw / 2) * w),
            min(float(h), (cy + bh / 2) * h),
        ]

    def _run_tiled(self, image, text, pos_boxes, neg_text, grid=2, overlap=0.12):
        """Pentru poze foarte aglomerate (modelul are max 900 detecții/trecere):
        împarte în grid x grid bucăți cu suprapunere și unește cu NMS."""
        from torchvision.ops import nms

        W, H = image.size
        tw, th = W / grid, H / grid
        ox, oy = tw * overlap, th * overlap
        all_b, all_s = [], []
        for gy in range(grid):
            for gx in range(grid):
                x0, y0 = max(0, gx * tw - ox), max(0, gy * th - oy)
                x1, y1 = min(W, (gx + 1) * tw + ox), min(H, (gy + 1) * th + oy)
                tile = image.crop((round(x0), round(y0), round(x1), round(y1)))
                b, s = self._run(tile, text, pos_boxes, neg_text, ex_image=image)
                if not len(b):
                    continue
                cw, ch = tile.size
                b = b.copy()
                b[:, 0] = (b[:, 0] * cw + round(x0)) / W
                b[:, 1] = (b[:, 1] * ch + round(y0)) / H
                b[:, 2] = b[:, 2] * cw / W
                b[:, 3] = b[:, 3] * ch / H
                all_b.append(b)
                all_s.append(s)
        if not all_b:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        b = np.concatenate(all_b)
        s = np.concatenate(all_s)
        t = torch.from_numpy(b)
        xyxy = torch.stack([t[:, 0] - t[:, 2] / 2, t[:, 1] - t[:, 3] / 2,
                            t[:, 0] + t[:, 2] / 2, t[:, 1] + t[:, 3] / 2], 1)
        keep = nms(xyxy, torch.from_numpy(s), 0.5).numpy()
        return b[keep], s[keep]

    def _pseudo_exemplar(self, boxes, scores, w, h):
        """Cel mai sigur box de mărime 'tipică' -> exemplu automat (pseudo-exemplar)."""
        good = scores >= DEFAULT_THRESH
        if not good.any():
            return None
        b = boxes[good]
        area = b[:, 2] * b[:, 3]
        med = np.median(area)
        # elimină box-uri uriașe (fundal / grup) și minuscule
        ok = (area < med * 3) & (area > med / 3) & (area < 0.25)
        idx = int(np.argmax(np.where(ok, scores[good], -1)))
        if not ok[idx]:
            return None
        return self._to_xyxy(b[idx], w, h)

    # ----------------------------------------------------------------- API
    def count(
        self,
        image: Image.Image,
        text: str = "",
        boxes_xyxy: list | None = None,
        neg_text: str = "",
        refine: bool = True,
    ) -> CountResult:
        """
        - text și/sau boxes: ce să numere
        - nimic: mod AUTOMAT -> numără tipul dominant de obiect din poză
        - refine: rulează a 2-a trecere cu pseudo-exemplar (metoda CountGD++)
        """
        image = image.convert("RGB")
        w, h = image.size
        text = (text or "").strip()
        boxes_xyxy = [list(map(float, b)) for b in (boxes_xyxy or [])]
        t0 = time.time()

        with self.lock:
            if boxes_xyxy:
                mode = "exemplu" + (" + text" if text else "")
                b, s = self._run(image, text, boxes_xyxy, neg_text)
                ex = boxes_xyxy[0]
            else:
                mode = "text" if text else "automat"
                first_text = text or AUTO_TEXT
                b, s = self._run(image, first_text, [], neg_text)
                ex = self._pseudo_exemplar(b, s, w, h) if refine else None
                if ex is not None:
                    # automat: doar exemplul vizual -> prinde exact "ce e mai mult" în poză
                    b2, s2 = self._run(image, text, [ex], neg_text)
                    if len(b2) and (s2 >= DEFAULT_THRESH).any():
                        b, s = b2, s2
                    mode += " + pseudo-exemplu"

            # saturat (aproape de limita de 900) -> rulează pe bucăți
            if int((s >= DEFAULT_THRESH).sum()) >= SATURATION:
                pos = boxes_xyxy or ([ex] if ex else [])
                b, s = self._run_tiled(image, text or ("" if pos else AUTO_TEXT), pos, neg_text)
                mode += " + bucăți"

        n = int((s >= DEFAULT_THRESH).sum())
        return CountResult(
            count=n,
            boxes=np.round(b, 5).tolist(),
            scores=np.round(s, 4).tolist(),
            mode=mode,
            exemplar=[round(v, 1) for v in ex] if ex else None,
            seconds=round(time.time() - t0, 2),
            device=str(self.device),
        )
