"""Hero-view consistency between the user's clicks and the object's name.

The other cameras find the object by its name alone (SAM 3.1 text prompt),
so before they do, the name is checked in the one view where the user's
intent is known: the clicked mask is compared with what the name finds in
the same image. A name that finds the clicked object is safe to carry to
the other views; a click mask that swallowed a neighbour (a bonsai and the
bicycle behind it) is narrowed to the named instance under the clicks.
"""
from __future__ import annotations

import re

import numpy as np

AGREE = "agree"
CLICK_TOO_BIG = "click_mask_too_big"      # the clicks grabbed more than the named object
CLICK_TOO_SMALL = "click_mask_too_small"  # the name covers more than was clicked
NAME_ELSEWHERE = "name_elsewhere"         # the name finds other things, not the clicked one
NAME_NOT_FOUND = "name_not_found"
DISAGREE = "disagree"


def clean_name(text: str, *, max_words: int = 5) -> str:
    """A model's answer as a short lowercase object name: the first line,
    without quotes, punctuation or a leading article."""
    line = next((l for l in str(text).strip().splitlines() if l.strip()), "")
    words = re.sub(r"[^a-z0-9 -]", " ", line.lower()).split()
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words[:max_words])


def parse_points(value) -> np.ndarray:
    """Comfy point JSON ([{"x":..,"y":..}, ...]) or [[x, y], ...] as an (N, 2) array."""
    import json
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    rows = [[p["x"], p["y"]] if isinstance(p, dict) else list(p)[:2] for p in value or []]
    return np.asarray(rows, np.float64).reshape(-1, 2)


def _hits(mask: np.ndarray, points: np.ndarray) -> int:
    if not len(points):
        return 0
    h, w = mask.shape
    x = np.clip(points[:, 0].astype(int), 0, w - 1)
    y = np.clip(points[:, 1].astype(int), 0, h - 1)
    return int(mask[y, x].sum())


def compare_hero_masks(click_mask, name_masks, positive, negative=(), *, min_iou: float = 0.6,
                       contain: float = 0.85) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Compare the click mask with the name's instance masks in the hero view.

    Returns the mask to carry on, the name's matched instances (None when
    the name found nothing) and a report. ``passed`` means the name and
    the clicks agree and the other views can be searched by the name without
    asking the user; otherwise ``status`` says what differs and the chosen
    mask is the best guess for the user to confirm."""
    click = np.asarray(click_mask) > 0.5
    if not click.any():
        raise ValueError("the click mask is empty")
    masks, resized = [], 0
    for m in (name_masks if name_masks is not None else []):
        m = np.asarray(m) > 0.5
        if m.shape != click.shape:
            # A segmenter may answer at another resolution than the view it
            # was shown; compared at the click mask's size, not dropped (which
            # would read as "name not found").
            from PIL import Image
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (click.shape[1], click.shape[0]), Image.NEAREST)) > 127
            resized += 1
        if m.any():
            masks.append(m)
    pos, neg = parse_points(positive), parse_points(negative)
    report = {"click_pixels": int(click.sum()), "name_instances": len(masks),
              "positive_clicks": int(len(pos)), "negative_clicks": int(len(neg))}
    if resized:
        report["resized_name_masks"] = resized
    if not masks:
        return click, None, {**report, "status": NAME_NOT_FOUND, "passed": False, "chosen": "click",
                       "advice": "the name finds nothing in the hero view: rename the object"}

    rows = []
    for index, mask in enumerate(masks):
        inter = int((mask & click).sum())
        union = int((mask | click).sum())
        rows.append({"index": index, "pixels": int(mask.sum()), "iou": inter / union,
                     "inside_click": inter / max(1, int(mask.sum())),   # share of the instance under the click mask
                     "covers_click": inter / max(1, int(click.sum())),  # share of the click mask it covers
                     "positive_hits": _hits(mask, pos), "negative_hits": _hits(mask, neg)})
    # The instances the user meant: those under include clicks (several when
    # several objects were clicked), else the one overlapping the clicks most.
    group = [r for r in rows if r["positive_hits"] > r["negative_hits"]]
    if not group:
        group = [max(rows, key=lambda r: (r["positive_hits"] - r["negative_hits"], r["iou"]))]
    chosen_name = np.logical_or.reduce([masks[r["index"]] for r in group])
    inter = int((chosen_name & click).sum())
    best = {"pixels": int(chosen_name.sum()), "iou": inter / int((chosen_name | click).sum()),
            "inside_click": inter / max(1, int(chosen_name.sum())), "covers_click": inter / int(click.sum()),
            "positive_hits": _hits(chosen_name, pos)}
    report.update(instances=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
                  matched_instances=[r["index"] for r in group], iou=round(best["iou"], 4),
                  name_pixels=best["pixels"])

    if best["iou"] >= min_iou:
        if (best["covers_click"] >= 0.95 and best["pixels"] > 1.1 * click.sum()
                and _hits(chosen_name, neg) == 0):
            # The clicked mask is a splat render's, with holes and missed
            # parts; the name's is the whole object around it. Playroom (23
            # Oct): the click mask left the buggy's seat out, the hull kept
            # it and Qwen, shown it in the removed render, painted it back.
            return click | chosen_name, chosen_name, {**report, "status": AGREE, "passed": True,
                                                      "chosen": "name"}
        return click, chosen_name, {**report, "status": AGREE, "passed": True, "chosen": "click"}
    if best["positive_hits"] == 0 and best["covers_click"] < 0.1:
        return click, chosen_name, {**report, "status": NAME_ELSEWHERE, "passed": False, "chosen": "click",
                       "advice": "the name finds other objects, not the clicked one: rename it"}
    if best["inside_click"] >= contain and best["pixels"] < click.sum() and best["positive_hits"]:
        # Bonsai + bicycle: the named instance under the clicks is part of the
        # click mask, and the rest of the click mask is something else.
        return chosen_name, chosen_name, {**report, "status": CLICK_TOO_BIG, "passed": False, "chosen": "name",
                             "advice": "the clicks also selected something else; using the named object "
                                       "(add an exclude click on the extra part to keep yours)"}
    if best["covers_click"] >= contain and best["pixels"] > click.sum():
        return click, chosen_name, {**report, "status": CLICK_TOO_SMALL, "passed": False, "chosen": "click",
                       "advice": "the name covers more than was clicked: name just the clicked part, "
                                 "or click the rest"}
    return click, chosen_name, {**report, "status": DISAGREE, "passed": False, "chosen": "click",
                   "advice": "the name and the clicks select different shapes: check both"}


def focus_crop(image: np.ndarray, mask: np.ndarray, *, pad: float = 0.25, dim: float = 0.35,
               size: int = 448, outline: int = 3) -> np.ndarray:
    """The region around a mask with everything else dimmed and the mask
    outlined, resized so its long side is ``size``: what a vision-language
    model is shown to name the selection."""
    from PIL import Image
    from scipy import ndimage

    rgb = np.asarray(image)[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    region = np.asarray(mask) > 0.5
    ys, xs = np.nonzero(region)
    if not ys.size:
        raise ValueError("cannot crop around an empty mask")
    border = int(pad * max(np.ptp(ys) + 1, np.ptp(xs) + 1)) + 8
    y0, y1 = max(0, ys.min() - border), min(region.shape[0], ys.max() + border + 1)
    x0, x1 = max(0, xs.min() - border), min(region.shape[1], xs.max() + border + 1)
    out = rgb.astype(np.float32)
    out[~region] *= dim
    if outline:
        edge = ndimage.binary_dilation(region, iterations=outline) & ~region
        out[edge] = (255, 220, 0)
    crop = Image.fromarray(out[y0:y1, x0:x1].round().astype(np.uint8))
    scale = size / max(crop.size)
    crop = crop.resize((max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))), Image.LANCZOS)
    return np.asarray(crop)


def overlay(image: np.ndarray, click_mask, chosen, name_mask=None) -> np.ndarray:
    """The hero view with the chosen selection tinted red, the click mask
    outlined in yellow and the name's instance outlined in cyan."""
    from scipy import ndimage

    rgb = np.asarray(image)[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    out = rgb.astype(np.float32)
    chosen = np.asarray(chosen) > 0.5
    out[chosen] = 0.5 * out[chosen] + 0.5 * np.array([215, 38, 61])
    for mask, colour, width in ((click_mask, (255, 220, 0), 3), (name_mask, (0, 220, 255), 2)):
        if mask is None:
            continue
        mask = np.asarray(mask) > 0.5
        out[ndimage.binary_dilation(mask, iterations=width) & ~mask] = colour
    return out.round().astype(np.uint8)
