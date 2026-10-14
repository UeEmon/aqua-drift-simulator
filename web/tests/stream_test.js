"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Run the real app transport with browser APIs replaced by controllable doubles.
const app = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
const start = app.indexOf("const stream = (() => {");
const end = app.indexOf("\nwindow.aquaDrift =", start);
assert.ok(start >= 0 && end > start, "app stream entry point exists");
const source = `${app.slice(start, end)}\nglobalThis.client = stream;`;

function browser(workerMode) {
  const sockets = [];
  const workers = [];
  const updates = [];
  const timers = [];
  const statuses = [];
  const scales = [];
  class Socket {
    constructor(url) { this.url = url; this.handlers = {}; sockets.push(this); }
    addEventListener(name, fn) { this.handlers[name] = fn; }
    emit(name, data) { this.handlers[name]?.(data); }
    close() { this.emit("close"); }
  }
  class Worker {
    constructor() {
      if (workerMode === "throws") throw new Error("worker blocked");
      this.messages = []; this.terminated = false; workers.push(this);
    }
    postMessage(message) {
      if (workerMode === "post-throws" || this.failMessages) throw new Error("worker stopped");
      this.messages.push(message);
    }
    terminate() { this.terminated = true; }
  }
  const context = {
    location: { protocol: "https:", host: "example.test:8090" },
    WebSocket: Socket,
    Worker: workerMode === "missing" ? undefined : Worker,
    AquaDecoder: { createDecoder: () => ({
      setExaggeration: (value) => { scales.push(value); return []; },
      decode: (message) => ({ snapshot: message, tracks: [], regionChanged: false }),
    }) },
    telemetry: { worker: false },
    performance: { now: () => 0 },
    exaggeration: () => 10,
    setText: (_, value) => statuses.push(value),
    setMessage: () => {},
    handleUpdate: (result) => updates.push(result),
    setTimeout: (fn) => { timers.push(fn); },
    window: { console: { error: () => {} } },
    console: { error: () => {} },
  };
  vm.createContext(context);
  return { context, sockets, workers, updates, timers, statuses, scales,
    start: () => vm.runInContext(source, context) };
}

test("normal reception works without Worker support and reconnects", () => {
  const b = browser("missing"); b.start();
  assert.equal(b.sockets.length, 1);
  assert.equal(b.sockets[0].url, "wss://example.test:8090/ws");
  b.sockets[0].emit("open");
  b.sockets[0].emit("message", { data: '{"tick":1}' });
  assert.equal(b.statuses.at(-1), "接続済み");
  assert.equal(b.updates.at(-1).snapshot.tick, 1);
  b.sockets[0].emit("close");
  b.timers.shift()();
  assert.equal(b.sockets.length, 2);
});

test("blocked Worker construction automatically starts normal reception", () => {
  const b = browser("throws");
  assert.doesNotThrow(b.start);
  assert.equal(b.sockets.length, 1);
  assert.equal(b.context.telemetry.worker, false);
});

test("Worker script failure falls back once and preserves the latest scale", () => {
  const b = browser("available"); b.start();
  const worker = b.workers[0];
  b.context.client.setExaggeration(25);
  assert.equal(typeof worker.onerror, "function");
  const onerror = worker.onerror;
  let prevented = 0;
  const error = { preventDefault: () => { prevented += 1; } };
  onerror(error);
  onerror(error); // a queued second error must not create a second socket
  assert.equal(prevented, 2);
  assert.equal(worker.terminated, true);
  assert.equal(b.context.telemetry.worker, false);
  assert.equal(b.sockets.length, 1);
  assert.equal(b.scales.at(-1), 25);
  b.sockets[0].emit("message", { data: '{"tick":2}' });
  assert.equal(b.updates.at(-1).snapshot.tick, 2);
  b.context.client.setExaggeration(5);
  assert.equal(b.scales.at(-1), 5);
});

test("working Worker keeps reception off the main thread", () => {
  const b = browser("available"); b.start();
  assert.equal(b.sockets.length, 0);
  assert.equal(b.context.telemetry.worker, true);
  const worker = b.workers[0];
  worker.onmessage({ data: { type: "status", connected: true } });
  worker.onmessage({ data: { type: "update", result: { snapshot: { tick: 3 } }, bytes: 10, parseMs: 1 } });
  assert.equal(b.statuses.at(-1), "接続済み");
  assert.equal(b.updates.at(-1).snapshot.tick, 3);
  b.context.client.setExaggeration(15);
  assert.equal(worker.messages.at(-1).value, 15);
});

test("failed initial Worker message terminates the worker and opens one socket", () => {
  const b = browser("post-throws");
  assert.doesNotThrow(b.start);
  assert.equal(b.workers[0].terminated, true);
  assert.equal(b.sockets.length, 1);
  assert.equal(b.context.telemetry.worker, false);
});

test("Worker message failure after startup preserves the requested scale", () => {
  const b = browser("available"); b.start();
  b.workers[0].failMessages = true;
  assert.doesNotThrow(() => b.context.client.setExaggeration(30));
  assert.equal(b.workers[0].terminated, true);
  assert.equal(b.sockets.length, 1);
  assert.equal(b.scales.at(-1), 30);
});
