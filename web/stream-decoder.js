/* AQUA-DRIFT stream decoder (protocol v2). Pure functions, no DOM / Cesium:
 * used inside the Web Worker (stream-worker.js), as a main-thread fallback, and in Node tests.
 *
 * Input:  compact delta messages from WebSocket /ws (see backend aqua_drift/wire.py)
 * Output: { snapshot, region, regionChanged, tracks: [{key, reset, keep, xyz}], stats }
 *   snapshot  - reconstructed snapshot in the familiar shape (estimates without track arrays)
 *   tracks    - per track (estimate modes, truth, observers): which points the renderer keeps
 *               (`keep`) and the new points as Earth-centred XYZ (Float64Array, WGS84,
 *               depth exaggeration applied) - ready for GPU upload without further math
 */
(function (root) {
  "use strict";

  const WGS84_A = 6378137.0;
  const WGS84_E2 = 6.69437999014e-3;
  const FT_TO_M = 0.3048;
  const DEG = Math.PI / 180;
  const MAX_ACCUMULATED_POINTS = 4000; // truth / observer tracks kept client-side
  const DROP_CHUNK = 1000;

  function ecef(lat, lon, depthFt, exaggeration, out, offset) {
    const phi = lat * DEG;
    const lambda = lon * DEG;
    const h = -depthFt * FT_TO_M * exaggeration;
    const s = Math.sin(phi);
    const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
    const c = Math.cos(phi);
    out[offset] = (n + h) * c * Math.cos(lambda);
    out[offset + 1] = (n + h) * c * Math.sin(lambda);
    out[offset + 2] = (n * (1 - WGS84_E2) + h) * s;
  }

  function toXyz(rows, start, exaggeration) {
    // rows: [[tick, lat, lon, depth, ...], ...]
    const count = rows.length - start;
    const out = new Float64Array(Math.max(count, 0) * 3);
    for (let i = 0; i < count; i += 1) {
      const r = rows[start + i];
      ecef(r[1], r[2], r[3], exaggeration, out, i * 3);
    }
    return out;
  }

  function createDecoder() {
    const st = {
      exaggeration: 10,
      runKey: null,
      config: null,
      cpa: [],
      current: null,
      deployment: { standby_count: 0, pending_placements: 0, last_deploy_tick: null, history: [] },
      archived: [],
      lloyd: null,
      region: null,
      est: {}, // mode -> rows
      truth: [], // [tick, lat, lon, depth]
      observers: {}, // id -> rows
    };

    function resetAccumulated() {
      st.truth = [];
      st.observers = {};
      st.est = {};
    }

    function appendAccumulated(key, rows, row, updates) {
      const last = rows[rows.length - 1];
      if (last && last[0] === row[0]) return rows;
      rows.push(row);
      if (rows.length > MAX_ACCUMULATED_POINTS) {
        rows.splice(0, DROP_CHUNK); // drop a whole block: one reset instead of per-point shifts
        updates.push({ key, reset: true, keep: 0, xyz: toXyz(rows, 0, st.exaggeration) });
      } else {
        updates.push({ key, reset: false, keep: rows.length - 1, xyz: toXyz(rows, rows.length - 1, st.exaggeration) });
      }
      return rows;
    }

    function decode(message) {
      const updates = [];
      const runKey = `${message.gen}-${message.ctl ? message.ctl.run_id : 0}`;
      if (message.gen !== st.gen) {
        resetAccumulated();
        updates.push({ key: "*", reset: true, keep: 0, xyz: new Float64Array(0) }); // clear all
        st.gen = message.gen;
      }
      if (runKey !== st.runKey) {
        st.runKey = runKey;
        st.est = {};
        for (const mode of ["ONLINE", "SMOOTHED"]) updates.push({ key: `est:${mode}`, reset: true, keep: 0, xyz: new Float64Array(0) });
      }
      if ("cfg" in message) st.config = message.cfg;
      if ("cpa" in message) st.cpa = message.cpa;
      if ("cur" in message) st.current = message.cur;
      if ("dep" in message) st.deployment = message.dep;
      if ("arch" in message) st.archived = message.arch;
      if ("lly" in message) st.lloyd = message.lly;
      let regionChanged = false;
      if ("rgn" in message) {
        st.region = message.rgn ? expandRegion(message.rgn) : null;
        regionChanged = true;
      }

      // truth track (accumulated client-side)
      const target = message.target;
      if (target) {
        const p = target.position;
        st.truth = appendAccumulated("truth", st.truth, [target.tick, p.latitude, p.longitude, p.depth_ft], updates);
      }

      // observers + their drift tracks
      const observers = message.obs.map(([id, lat, lon, depth, registered, last]) => {
        const rows = st.observers[id] || [];
        st.observers[id] = appendAccumulated(`obs:${id}`, rows, [last, lat, lon, depth], updates);
        return {
          state: { observer_id: id, tick: last, position: { latitude: lat, longitude: lon, depth_ft: depth } },
          registered_tick: registered,
          last_tick: last,
        };
      });

      // estimates (tracks as diffs)
      const estimates = (message.est || []).map((body) => {
        const mode = body.mode;
        if (body.trk) {
          const rows = st.est[mode] || [];
          if (body.trk.reset) {
            st.est[mode] = body.trk.pts;
            updates.push({ key: `est:${mode}`, reset: true, keep: 0, xyz: toXyz(st.est[mode], 0, st.exaggeration) });
          } else {
            const keep = Math.min(body.trk.keep, rows.length);
            rows.length = keep;
            for (const pt of body.trk.pts) rows.push(pt);
            st.est[mode] = rows;
            updates.push({ key: `est:${mode}`, reset: false, keep, xyz: toXyz(rows, keep, st.exaggeration) });
          }
        }
        const estimate = Object.assign({}, body);
        delete estimate.trk;
        delete estimate.rel;
        estimate.relative = (body.rel || []).map(([id, rs, sr, det]) => ({
          observer_id: id, relative_speed_kt: rs, slant_range_yd: sr, detected: det,
        }));
        estimate.track_length = (st.est[mode] || []).length;
        const rows = st.est[mode] || [];
        estimate.last_track_tick = rows.length ? rows[rows.length - 1][0] : null;
        return estimate;
      });
      if (!(message.est || []).length && Object.keys(st.est).length) {
        st.est = {};
        for (const mode of ["ONLINE", "SMOOTHED"]) updates.push({ key: `est:${mode}`, reset: true, keep: 0, xyz: new Float64Array(0) });
      }

      const dop = message.dop;
      const snapshot = {
        tick: message.t,
        generation: message.gen,
        estimation: message.ctl,
        config: st.config,
        target,
        observers,
        doppler: dop
          ? {
            tick: dop.t,
            observations: dop.det.map((id) => ({ observer_id: id, detected: true })),
            truth: dop.tr.map(([id, rs, sr, tb, tick]) => ({
              observer_id: id, relative_speed_kt: rs, slant_range_yd: sr, true_bearing_deg: tb, tick,
            })),
          }
          : null,
        bearings: (message.brg || []).map(([id, tick, b, lat, lon, depth]) => ({
          observer_id: id, tick, bearing_deg: b, observer_position: { latitude: lat, longitude: lon, depth_ft: depth },
        })),
        estimates,
        cpa: st.cpa,
        current_estimate: st.current,
        deployment: st.deployment,
        archived_observer_ids: st.archived,
        lloyd: st.lloyd,
      };
      return { snapshot, region: regionChanged ? st.region : undefined, regionChanged, tracks: updates };
    }

    function expandRegion(r) {
      return {
        probability_pct: r.pct,
        disconnected: r.disc,
        components: r.c.map((c) => ({
          probability_mass_pct: c.m,
          centroid: { latitude: c.cen[0], longitude: c.cen[1], depth_ft: c.cen[2] },
          polygon: c.poly,
          min_depth_ft: c.zmin,
          max_depth_ft: c.zmax,
          voxels: c.vox,
          voxel_size_yd: c.vs,
          voxel_height_ft: c.vh,
        })),
      };
    }

    function setExaggeration(value) {
      // recompute every track with the new vertical scale
      st.exaggeration = value;
      const updates = [{ key: "truth", reset: true, keep: 0, xyz: toXyz(st.truth, 0, value) }];
      for (const [id, rows] of Object.entries(st.observers)) updates.push({ key: `obs:${id}`, reset: true, keep: 0, xyz: toXyz(rows, 0, value) });
      for (const [mode, rows] of Object.entries(st.est)) updates.push({ key: `est:${mode}`, reset: true, keep: 0, xyz: toXyz(rows, 0, value) });
      return updates;
    }

    function trackRows(key) {
      if (key === "truth") return st.truth;
      if (key.startsWith("obs:")) return st.observers[key.slice(4)] || [];
      if (key.startsWith("est:")) return st.est[key.slice(4)] || [];
      return [];
    }

    return { decode, setExaggeration, trackRows, ecef };
  }

  root.AquaDecoder = { createDecoder, ecef };
  if (typeof module !== "undefined" && module.exports) module.exports = root.AquaDecoder;
})(typeof self !== "undefined" ? self : globalThis);
