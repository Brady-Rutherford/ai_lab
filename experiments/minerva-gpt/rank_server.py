"""Local ranking page: judge the model's answers blind; the picks become DPO pairs.

    uv run python rank_server.py --a data/generations/A-val.jsonl --b data/generations/B-val.jsonl
    then open http://127.0.0.1:8765

Rank: one held-out ask and its answers (2 from A, 2 from B), shuffled and unlabeled. For each,
say "sounds like me" yes or no, then pick the best of the yeses. Saved to data/rankings.jsonl with
the variants recorded, so the same picks also show whether classmates' data helped.
Blind test: my real answer next to one model answer; guess which is mine (data/blind.jsonl).
50% correct means the model is indistinguishable from me on that ask.

Binds to 127.0.0.1 only, and the page loads nothing from the internet: the answers are class data.
"""

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import common

HOST, PORT = "127.0.0.1", 8765
PAGE = Path(__file__).with_name("rank.html")


def read(path):
    return common.read_jsonl(path) if path.exists() else []


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Store:
    def __init__(self, paths, data):
        self.asks = {}
        for path in paths:
            for row in common.read_jsonl(path):
                ask = self.asks.setdefault(row["id"], {"row": row, "answers": []})
                ask["answers"] += [{"text": s, "variant": row["variant"]} for s in row["samples"]]
        self.files = {"rank": data / "rankings.jsonl", "blind": data / "blind.jsonl"}
        self.done = {mode: {r["id"] for r in read(path)} for mode, path in self.files.items()}
        self.order = sorted(self.asks)
        random.Random(0).shuffle(self.order)
        self.shown = {}  # (mode, id) -> what the page was shown, with the hidden labels

    def next(self, mode):
        ask_id = next((i for i in self.order if i not in self.done[mode]), None)
        if ask_id is None:
            return {"done": True, **self.stats()}
        ask, rng = self.asks[ask_id], random.Random(f"{mode}:{ask_id}")
        if mode == "rank":
            options = ask["answers"][:]
        else:
            model = rng.choice(ask["answers"])
            options = [{"text": ask["row"]["real"], "variant": "real"}, model]
        rng.shuffle(options)
        self.shown[(mode, ask_id)] = options
        return {"id": ask_id, "prompt": ask["row"]["prompt"][0]["content"],
                "answers": [o["text"] for o in options], **self.stats()}

    def save(self, mode, body):
        ask_id = body["id"]
        options, row = self.shown.pop((mode, ask_id)), self.asks[ask_id]["row"]
        base = {"id": ask_id, "class_id": row["class_id"], "source": row["source"]}
        if mode == "rank":
            append(self.files[mode], {**base, "prompt": row["prompt"], "answers": options,
                                      "voice": body["voice"], "best": body.get("best")})
        else:
            guess = options[body["guess"]]
            model = next(o for o in options if o["variant"] != "real")
            append(self.files[mode], {**base, "variant": model["variant"],
                                      "correct": guess["variant"] == "real"})
        self.done[mode].add(ask_id)

    def stats(self):
        blind = read(self.files["blind"])
        rate = {v: [r["correct"] for r in blind if r["variant"] == v] for v in ("A", "B")}
        return {"ranked": len(self.done["rank"]), "total": len(self.asks),
                "blind": {v: {"n": len(x), "correct": sum(x)} for v, x in rate.items()}}


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/next":
                mode = parse_qs(url.query).get("mode", ["rank"])[0]
                self.reply(200, store.next("blind" if mode == "blind" else "rank"))
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            mode = {"/api/rank": "rank", "/api/blind": "blind"}.get(self.path)
            if mode is None:
                return self.reply(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            store.save(mode, body)
            self.reply(200, {"ok": True})

        def log_message(self, *args):  # keep class text out of the terminal log
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, help="generations from variant A")
    ap.add_argument("--b", type=Path, help="generations from variant B")
    ap.add_argument("--data", type=Path, default=common.DATA)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    store = Store([p for p in (args.a, args.b) if p], args.data)
    print(f"{len(store.asks)} asks loaded; open http://{HOST}:{args.port}")
    ThreadingHTTPServer((HOST, args.port), handler(store)).serve_forever()


if __name__ == "__main__":
    main()
