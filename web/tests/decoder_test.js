// Decoder test: replays a Python-encoded protocol-v2 stream through web/stream-decoder.js and
// checks that the reconstructed tracks equal the server-side tracks at every checkpoint, that
// the XYZ coordinates are consistent, and reports the payload reduction.
// usage: node web/tests/decoder_test.js fixture.json
"use strict";
const fs = require("fs");
const path = require("path");
const { createDecoder, ecef } = require(path.join(__dirname, "..", "stream-decoder.js"));

const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const decoder = createDecoder();
decoder.setExaggeration(10);
const checkpoints = new Map(fixture.checkpoints.map((c) => [c.index, c]));
const xyzTracks = new Map(); // key -> array of numbers (renderer view)
let failures = 0;
const fail = (m) => { failures += 1; console.log("FAIL", m); };

fixture.messages.forEach((raw, index) => {
  const out = decoder.decode(JSON.parse(raw));
  for (const t of out.tracks) {
    if (t.key === "*") { xyzTracks.clear(); continue; }
    const arr = t.reset ? [] : (xyzTracks.get(t.key) || []).slice(0, t.keep * 3);
    for (const v of t.xyz) arr.push(v);
    xyzTracks.set(t.key, arr);
  }
  const cp = checkpoints.get(index);
  if (!cp) return;
  for (const [mode, rows] of Object.entries(cp.tracks)) {
    const got = decoder.trackRows(`est:${mode}`);
    if (JSON.stringify(got) !== JSON.stringify(rows)) fail(`${mode} rows differ at message ${index}`);
    const xyz = xyzTracks.get(`est:${mode}`) || [];
    if (xyz.length !== rows.length * 3) { fail(`${mode} xyz length ${xyz.length} != ${rows.length * 3}`); continue; }
    const probe = new Float64Array(3);
    rows.forEach((r, i) => {
      ecef(r[1], r[2], r[3], 10, probe, 0);
      for (let k = 0; k < 3; k += 1) if (Math.abs(probe[k] - xyz[i * 3 + k]) > 1e-6) fail(`${mode} xyz mismatch at ${i}`);
    });
  }
  const est = out.snapshot.estimates[0];
  if (cp.tracks.ONLINE && est.track_length !== cp.tracks.ONLINE.length) fail(`track_length mismatch at ${index}`);
  if (out.snapshot.observers.length !== cp.observers) fail(`observer count at ${index}`);
});

// truth track accumulated client-side: one point per tick
const truthRows = decoder.trackRows("truth");
if (truthRows.length !== fixture.messages.length) fail(`truth rows ${truthRows.length} != ${fixture.messages.length}`);

const warm = Math.floor(fixture.messages.length / 2);
const delta = fixture.messages.slice(warm).reduce((a, m) => a + m.length, 0) / (fixture.messages.length - warm);
const full = fixture.full_sizes.slice(warm).reduce((a, b) => a + b, 0) / (fixture.full_sizes.length - warm);
console.log(`messages ${fixture.messages.length}, checkpoints ${fixture.checkpoints.length}`);
console.log(`payload per update: delta ${(delta / 1024).toFixed(1)} KB vs full ${(full / 1024).toFixed(1)} KB (${(full / delta).toFixed(1)}x smaller)`);
if (failures) { console.log(`${failures} failures`); process.exit(1); }
console.log("DECODER OK");
