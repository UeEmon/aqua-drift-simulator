"""Label decluttering checks shared by the UI smoke test and the full-stack UI check.

The labels (添字) on the map must not overlap one another, must not sit under the map panels
(legend, camera readout, toolbar, wind/force and layer panels, status strip) and must stay inside
the 3D view. LABELS_JS measures the labels as placed in the last frame against the panels' DOM
rectangles and returns the violations.
"""
from __future__ import annotations

LABELS_JS = """() => {
  const a = window.aquaDrift; const d = a.declutter; const scene = a.viewer.scene;
  const c = scene.canvas.getBoundingClientRect();
  const panels = [...document.querySelectorAll(d.panels)]
    .filter((e) => !e.closest('[hidden]')).map((e) => [e, e.getBoundingClientRect()])
    .filter(([, r]) => r.width >= 1 && r.height >= 1)
    .map(([e, r]) => ({ name: e.id || e.className, x0: r.left - c.left, y0: r.top - c.top, x1: r.right - c.left, y1: r.bottom - c.top }));
  // the placed boxes carry a 2 px margin: touching labels are fine, overlapping ones are not
  const boxes = d.placed.map((p) => ({ text: p.label.text, x0: p.x0 + 2, y0: p.y0 + 2, x1: p.x1 - 2, y1: p.y1 - 2,
    moved: p.moved, shown: p.label._show, ox: p.label.pixelOffset.x, oy: p.label.pixelOffset.y }));
  const hit = (p, q) => p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1;
  const pairs = []; const underPanel = []; const outside = []; const notShown = [];
  boxes.forEach((p, i) => {
    if (!p.shown) notShown.push(p.text);
    for (let j = i + 1; j < boxes.length; j += 1) if (hit(p, boxes[j])) pairs.push(`${p.text} / ${boxes[j].text}`);
    for (const q of panels) if (hit(p, q)) underPanel.push(`${p.text} under ${q.name}`);
    if (p.x0 < 0 || p.y0 < 0 || p.x1 > c.width || p.y1 > c.height) outside.push(p.text);
  });
  return { stats: d.stats, enabled: d.enabled, placed: boxes.length, moved: boxes.filter((b) => b.moved).length,
           pairs, underPanel, outside, notShown };
}"""


def label_violations(result: dict) -> list[str]:
    out = []
    if result["pairs"]:
        out.append(f"labels overlap: {result['pairs'][:5]}")
    if result["underPanel"]:
        out.append(f"labels under map panels: {result['underPanel'][:5]}")
    if result["outside"]:
        out.append(f"labels outside the view: {result['outside'][:5]}")
    if result["notShown"]:
        out.append(f"placed labels not shown: {result['notShown'][:5]}")
    return out
