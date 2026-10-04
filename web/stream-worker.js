/* AQUA-DRIFT stream worker: WebSocket, JSON parsing, delta merge and coordinate conversion run
 * here, off the main (rendering) thread. Track coordinates are transferred as Float64Arrays
 * (zero-copy) so the main thread only hands them to the GPU. */
/* global AquaDecoder */
"use strict";
importScripts("/stream-decoder.js");

const decoder = AquaDecoder.createDecoder();
let socket = null;
let url = null;

function post(result, bytes, parseMs) {
  const transfer = result.tracks.map((t) => t.xyz.buffer);
  self.postMessage({ type: "update", result, bytes, parseMs }, transfer);
}

function connect() {
  socket = new WebSocket(url);
  socket.addEventListener("open", () => self.postMessage({ type: "status", connected: true }));
  socket.addEventListener("message", (event) => {
    const t0 = performance.now();
    let result;
    try {
      result = decoder.decode(JSON.parse(event.data));
    } catch (error) {
      self.postMessage({ type: "error", message: String(error && error.message || error) });
      return;
    }
    post(result, event.data.length, performance.now() - t0);
  });
  socket.addEventListener("close", () => {
    self.postMessage({ type: "status", connected: false });
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

self.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg.type === "connect") {
    url = msg.url;
    decoder.setExaggeration(msg.exaggeration || 10);
    connect();
  } else if (msg.type === "exaggeration") {
    const tracks = decoder.setExaggeration(msg.value);
    self.postMessage({ type: "update", result: { snapshot: null, tracks, regionChanged: false }, bytes: 0, parseMs: 0 },
      tracks.map((t) => t.xyz.buffer));
  }
});
