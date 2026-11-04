"""The context test's own smoke test: its logic and every failure path, with no GPU and no class data.

    uv run python smoke_context.py [dir]        # dir defaults to a fresh temp folder

Fake data and a tiny random OLMo 3 (8 layers, window of 32 tokens), so long prompts are short here.
The real vllm_sample.py, eval_context.py and pod scripts run against pod/fake_vllm, which scores
through plain Transformers. The pod is a folder with its own HOME, and ssh is a local stand-in.
This proves the plumbing and the checks; only the crosscheck on the pod can prove vLLM itself.
"""

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import common
import eval_context as ec

HERE = Path(__file__).resolve().parent
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def fake_data(data):
    rows, history = [], []
    words = "evidence sample method model reading class assumption result bias variance".split()
    for c in range(6):
        for i in range(5):
            content = (f"{common.TAG_ME}\nClass in CS156 - fake {c}\n\nReadings and pre-class instructions: "
                       f"{words[c]} and {words[i]}.\n\nProfessor: What about {words[(c + i) % 10]}?\nStudent 1: Maybe.")
            prompt = [{"role": "user", "content": content}]
            rows.append({"prompt": prompt, "completion": [{"role": "assistant", "content":
                         f"So I think the {words[i]} matters, like the {words[c]} too."}],
                         "source": "class_turn", "speaker": common.ME, "class_id": c, "los": []})
            if i == 2:  # not a spoken answer: the test must skip it, and keep its place for the baseline
                rows.append({**rows[-1], "source": "pcw"})
            lines = [f"Student {k % 3}: The {words[k % 10]} seems related to the {words[(k + c) % 10]}." for k in range(i * 6)]
            history.append({"id": common.prompt_key(prompt), "class_id": c, "history": "\n".join(lines)})
    (data / "sft").mkdir(parents=True)
    (data / "context").mkdir(parents=True)
    common.write_jsonl(data / "sft/val.jsonl", rows)
    common.write_jsonl(data / "context/val.jsonl", history)


def tiny_model(path):
    from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM

    tok = AutoTokenizer.from_pretrained(common.BASE_MODEL)
    config = Olmo3Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=8, num_attention_heads=2,
                         num_key_value_heads=2, intermediate_size=64, sliding_window=32,
                         max_position_embeddings=65536, initializer_range=0.2)
    Olmo3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)


def unit_tests():
    print("== logic")
    c = f"{common.TAG_ME}\nheader\n\nProfessor: ask?\nStudent 1: reply"
    mid, top = ec.insert_history(c, "A: b"), ec.insert_history(c, "A: b", top=True)
    check("every placement leaves the prompt's end as trained", all(x.endswith("Professor: ask?\nStudent 1: reply") for x in (c, mid, top)))
    check("before-the-ask keeps the voice tag first", mid.startswith(common.TAG_ME) and mid.index("Class so far:") < mid.index("Professor:"))
    check("at-the-top leaves the trained prompt whole after the class", top == f"Class so far:\nA: b\n\n{c}")
    check("no class so far leaves the prompt untouched", ec.insert_history(c, "") == c == ec.insert_history(c, "", top=True))

    base = {"ci95": [-0.01, 0.01], "habit_shift": {h: {"ci95": [-1, 1]} for h in ec.HABITS},
            "length_vs_window_ci95": [-0.05, 0.05], "speaker_vs_window_ci95": [-0.01, 0.01]}
    flag = lambda n: {h: {"ci95": [0.1, 2] if k < n else [-1, 1]} for k, h in enumerate(ec.HABITS)}  # noqa: E731
    cases = [("inside every limit", {}, "PASS"), ("loss interval crosses the limit", {"ci95": [0.0, 0.03]}, "UNSURE"),
             ("loss clearly worse", {"ci95": [0.02, 0.04]}, "FAIL"), ("loss clearly better", {"ci95": [-0.06, -0.02]}, "PASS"),
             ("one habit further", {"habit_shift": flag(1)}, "UNSURE"), ("two habits further", {"habit_shift": flag(2)}, "FAIL"),
             ("answers much longer", {"length_vs_window_ci95": [0.12, 0.3]}, "FAIL"),
             ("answers much shorter", {"length_vs_window_ci95": [-0.3, -0.12]}, "FAIL"),
             ("length interval crosses the band", {"length_vs_window_ci95": [0.02, 0.14]}, "UNSURE"),
             ("writes other speakers more", {"speaker_vs_window_ci95": [0.03, 0.08]}, "FAIL"),
             ("other-speaker interval crosses the limit", {"speaker_vs_window_ci95": [0.01, 0.05]}, "UNSURE")]
    for name, change, want in cases:
        got = ec.verdict({**base, **change})["verdict"]
        check(f"verdict: {name} -> {want}", got == want, got if got != want else "")

    like = ec.HABITS["like"]
    same = [["I like this, like a lot"], ["no habit here at all"]]
    point, lo, hi = ec.habit_shift(like, 100.0, same, same, [0, 1])
    check("habit shift: identical answers show no shift", point == 0 and lo == 0 and hi == 0)
    more = [["like like like like"], ["like like like like"]]
    check("habit shift: further from my rate is positive", ec.habit_shift(like, 0.0, more, same, [0, 1])[0] > 0)
    check("habit shift: closer to my rate is negative", ec.habit_shift(like, 1000.0, more, same, [0, 1])[0] < 0)

    lo, hi = ec.class_bootstrap([0.1] * 10, list(range(10)))
    check("bootstrap: a constant difference has no spread", math.isclose(lo, 0.1) and math.isclose(hi, 0.1))
    lo, hi = ec.class_bootstrap([0.0] * 100 + [1.0], [0] * 100 + [1])
    check("bootstrap: resamples whole classes, not answers", hi == 1.0, f"high end {hi:.3f}; by answers it would be near 0.03")
    ids = {"full": [[0] * n for n in (10, 100, 50, 200, 5)]}
    ids |= {"full-top": ids["full"], "window": [[1] * 3] * 5, "tail": [[1] * 4] * 5}
    items = ec.crosscheck_items(ids, [[9]] * 5, 5)
    check("crosscheck always includes the longest prompt", any(len(it["prompt"]) == 200 for it in items))
    check("crosscheck covers all four kinds of prompt", {it["variant"] for it in ec.crosscheck_items(ids, [[9]] * 5, 10)} == set(ec.VARIANTS))


def run(args, env, label=None, expect=0):
    r = subprocess.run(args, env=env, capture_output=True, text=True, cwd=env.get("_CWD", HERE))
    if label:
        check(label, r.returncode == expect, "" if r.returncode == expect else f"exit {r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}")
    return r


def calls(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []


def end_to_end(sy, model, env):
    print("== eval_context.py and vllm_sample.py, through the stand-in")
    data, out, log = sy / "data", sy / "out", sy / "vllm.log"
    base = [sys.executable, "eval_context.py", "--model", str(model), "--data", str(data), "--out", str(out),
            "--tail-tokens", "16", "--baseline", str(sy / "perrow.jsonl")]
    env = {**env, "FAKE_VLLM_LOG": str(log)}

    r = run(base + ["--crosscheck", "8"], env, "crosscheck passes when vLLM follows the window")
    res = json.loads((out / "context-crosscheck.json").read_text())
    check("crosscheck: the two agree in every group of prompts", len(res["groups"]) == 3 and all(g["mean_abs_diff"] < 1e-3 for g in res["groups"].values()), str({k: g["items"] for k, g in res["groups"].items()}))
    check("crosscheck: greedy answers match, first pass and from the cache", res["greedy_agreement"]["first_pass"] == 1 and res["greedy_agreement"]["from_prefix_cache"] == 1)
    check("crosscheck: the control shows it could notice", res["control_mean_abs_diff"] > ec.CROSSCHECK_LIMIT, f"{res['control_mean_abs_diff']:.4f}")
    r = run(base + ["--crosscheck", "8"], {**env, "FAKE_VLLM_NO_WINDOW": "1"}, "crosscheck exits 3 when vLLM ignores the window", expect=3)
    check("crosscheck says so plainly", "disagree" in r.stdout)
    r = run(base + ["--crosscheck", "8"], {**env, "FAKE_VLLM_LONG_FAULT": "-0.006"}, "crosscheck exits 3 on a fault in long prompts only", expect=3)
    res = json.loads((out / "context-crosscheck.json").read_text())
    check("crosscheck: that fault passes on all prompts together, and only the long group catches it",
          res["groups"]["all"]["passed"] and not res["groups"]["over 64"]["passed"] and not res["passed"],
          f"all {res['groups']['all']['mean_diff']:+.4f}, long {res['groups']['over 64']['mean_diff']:+.4f}, limit {ec.CROSSCHECK_BIAS}")
    run(base + ["--crosscheck", "8"], {**env, "FAKE_VLLM_CACHE_FAULT": "1"}, "crosscheck exits 3 when the prefix cache goes wrong", expect=3)
    run(base + ["--crosscheck", "0"], env, "crosscheck refuses too few prompts", expect=1)
    log.unlink(missing_ok=True)

    r = run(base, {**env, "FAKE_VLLM_NAN": "1"}, "a NaN score stops the run", expect=1)
    check("and says why", "NaN" in r.stderr)
    check("and isn't saved as finished", not list(out.glob("context-scores-*.json")))
    for f in out.glob("context-scores-*"):
        f.unlink()
    log.unlink(missing_ok=True)

    r = run(base, {**env, "FAKE_VLLM_CRASH_AFTER": "1"}, "a crash mid-scoring stops the run", expect=1)
    partial = list(out.glob("context-scores-*.partial"))
    check("the finished chunk survived the crash", len(partial) == 1 and len(json.loads(partial[0].read_text())) == 64)
    check("nothing pretends the run finished", not (out / "context-test.json").exists())
    first = calls(log)[0]
    log.unlink()

    r = run(base, env, "a rerun finishes")
    print("\n".join("     " + x for x in r.stdout.splitlines()[-32:]))
    after = [c for c in calls(log) if c["scoring"]]
    check("the rerun skipped the finished chunk", len(after) == 1 and after[0]["prompts"] == 120 - 64, f"{after}")
    seeds = [c for c in calls(log) if not c["scoring"]][0]["seeds"]
    check("each answer has its own seed", set(seeds[:30]) == set(range(30)), f"{len(set(seeds))} seeds")
    check("and the same one under each prompt", seeds == list(range(30)) * 4)
    check("longest prompts were scored first", first["shortest"] >= after[0]["longest"])
    log.unlink()

    keep = (out / "context-test-answers.jsonl").read_text()
    r = run(base, env, "a third run works")
    check("the third run reused the scores", not [c for c in calls(log) if c["scoring"]] and "earlier run of this exact job" in r.stdout)
    check("and its numbers are identical", (out / "context-test-answers.jsonl").read_text() == keep)

    rows = common.read_jsonl(out / "context-test-answers.jsonl")
    report = json.loads((out / "context-test.json").read_text())
    check("every answer has its loss under all four prompts", len(rows) == 30 and all(set(x["loss"]) == set(ec.VARIANTS) for x in rows))
    tiny = common.load_model(name=str(model), attn="sdpa").eval()
    from transformers import AutoTokenizer
    from generate import prompt_ids

    tok = AutoTokenizer.from_pretrained(str(model))
    tok.chat_template = tok.chat_template or common.CHAT_TEMPLATE
    val_all = common.read_jsonl(data / "sft/val.jsonl")
    val = [v for v in val_all if v["source"] == "class_turn"]
    history = {h["id"]: h["history"] for h in common.read_jsonl(data / "context/val.jsonl")}
    prompts = ec.build_prompts(tok, val, history, 16, 28000)
    answers = [tok(r["completion"][0]["content"] + tok.eos_token, add_special_tokens=False).input_ids for r in val]
    for v in ec.VARIANTS:
        direct = ec.hf_means(tiny, [{"prompt": prompt_ids(tok, p), "answer": a} for p, a in zip(prompts[v], answers)])
        gap = max(abs(-d - x["loss"][v]) for d, x in zip(direct, rows))
        check(f"{v}: each loss belongs to its own answer, after sorting and unsorting", gap < 1e-3, f"largest gap {gap:.6f}")
    check("full-top: the class never changes what comes after it", all(
        p[0]["content"].endswith(q[0]["content"]) for p, q in zip(prompts["full-top"], prompts["window"])))
    empty = [x for x, h in zip(rows, val) if not history[common.prompt_key(h["prompt"])]]
    check("an answer with no class so far scores the same under every prompt", len(empty) == 6 and all(
        max(x["loss"].values()) - min(x["loss"].values()) < 1e-3 for x in empty))

    V = report["variants"]
    check("the report covers all four variants", set(V) == set(ec.VARIANTS))
    check("every compared variant gets a verdict", all(V[v]["decision"]["verdict"] in ("PASS", "UNSURE", "FAIL") for v in ec.VARIANTS[1:]))
    check("the window has no comparison with itself", "ci95" not in V["window"] and "decision" not in V["window"])
    check("the length table is there", report["by_length"] and sum(b["answers"] for b in report["by_length"].values()) == 30)
    check("a file per variant for the rank page", all((out / f"B-val-ctx-{v}.jsonl").exists() for v in ec.VARIANTS))
    wins = sum(x["loss"]["tail"] < x["loss"]["window"] for x in rows)
    moved = sum(x["loss"]["tail"] != x["loss"]["window"] for x in rows)
    check("ties don't count against a variant's win share", moved < 30 and math.isclose(V["tail"]["better_on_share"], wins / moved))

    spoken = iter(x["loss"]["window"] for x in rows)  # every row of val.jsonl; only spoken answers can match
    perrow = [{"model": "x/final-dpobest-B", "losses": [next(spoken) if v["source"] == "class_turn" else 99.0 for v in val_all]}]
    common.write_jsonl(sy / "perrow.jsonl", perrow)
    r = run(base, env)
    check("baseline: matches row by row, skipping answers that aren't spoken", "mean difference 0.0000" in r.stdout and "larger than expected" not in r.stdout)
    perrow[0]["losses"] = [x + 0.05 for x in perrow[0]["losses"]]
    common.write_jsonl(sy / "perrow.jsonl", perrow)
    r = run(base, env)
    check("baseline: different earlier numbers raise a warning", "larger than expected" in r.stdout)
    (sy / "perrow.jsonl").unlink()


def shims(bin_dir):
    bin_dir.mkdir(exist_ok=True)
    scripts = {"ssh": '#!/bin/sh\nwhile :; do case "$1" in -p|-o|-i) shift 2;; *) break;; esac; done\nshift\n'
                      'cd "$FAKE_POD_HOME" && HOME="$FAKE_POD_HOME" exec /bin/bash -c "$*"\n',
               "setsid": '#!/bin/sh\nexec "$@"\n'}
    if not shutil.which("sha256sum"):
        scripts["sha256sum"] = '#!/bin/sh\nexec shasum -a 256 "$@"\n'
    for name, text in scripts.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)


def new_pod(sy, name, model):
    home = sy / name
    code = home / "brady-gpt"
    code.mkdir(parents=True)
    for f in HERE.glob("*.py"):
        shutil.copy(f, code)
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy(HERE / f, code)
    shutil.copytree(HERE / "pod", code / "pod", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(sy / "data", code / "data")
    (home / "models").mkdir()
    shutil.copytree(model, home / "models/dpobest-B")
    return home


def pod_scripts(sy, model, env):
    print("== pod/run_context_test.sh and pod/pull.sh, on a pretend pod")
    shims(sy / "bin")
    digest = subprocess.run(["shasum", "-a", "256", str(model / "model.safetensors")], capture_output=True, text=True).stdout.split()[0]
    pod_env = {**env, "PATH": f"{sy / 'bin'}:{env['PATH']}", "UV_PROJECT_ENVIRONMENT": str(HERE / ".venv"), "UV_NO_SYNC": "1",
               "UV_OFFLINE": "1", "UV_CACHE_DIR": str(Path.home() / ".cache/uv"), "MODEL_SHA": digest,
               "TAIL": "16", "CROSSCHECK": "6", "HF_TOKEN": "not-a-real-token"}

    def script(home, **extra):
        e = {**pod_env, "HOME": str(home), "_CWD": str(home), **extra}
        return e

    home = new_pod(sy, "pod-a", model)
    out, log = home / "brady-gpt/data/generations/7b", home / "fake.log"
    r = run(["bash", "brady-gpt/pod/run_context_test.sh"], script(home, LIMIT="12", FAKE_VLLM_REQUIRE_BACKEND="TRITON_ATTN", FAKE_VLLM_LOG=str(log)),
            "the default backend failing falls back to Triton, and the quick pass finishes")
    text = r.stdout + r.stderr
    check("the fallback really asks vLLM for Triton", {c["backend"] for c in calls(log)} >= {"TRITON_ATTN"}, str({c["backend"] for c in calls(log)}))
    check("the backend that passed is written down", (out / "context-backend.txt").read_text().strip() == "TRITON_ATTN")
    check("the quick pass says it was one", "quick pass of 12" in text and "Don't read anything into the verdicts" in text)
    check("its results stay in their own folder", (out / "quick/context-test.json").exists() and not (out / "context-test.json").exists())
    check("a finished run leaves no marker", not (out / "context-test.running").exists())
    check("the login is dropped before the run starts", calls(log) and not any(c["hf_token"] for c in calls(log)))

    r = run(["bash", "brady-gpt/pod/run_context_test.sh"], script(home, FAKE_VLLM_REQUIRE_BACKEND="TRITON_ATTN", HF_TOKEN=""), "a full run after it needs no login and reuses the backend")
    check("it skips the crosscheck", "is vLLM right" not in r.stdout)
    check("it runs every answer", json.loads((out / "context-test.json").read_text())["answers"] == 30)

    home = new_pod(sy, "pod-b", model)
    r = run(["bash", "brady-gpt/pod/run_context_test.sh"], script(home, FAKE_VLLM_NO_WINDOW="1"), "vLLM disagreeing stops the run with exit 3", expect=3)
    check("no fallback was tried, since the backend ran fine", "trying TRITON" not in r.stdout)
    check("the failure is stated", "context test FAILED" in r.stdout)
    check("no test ran", not (home / "brady-gpt/data/generations/7b/context-test.json").exists())
    check("the failed run stays marked as unfinished", (home / "brady-gpt/data/generations/7b/context-test.running").exists())

    home = new_pod(sy, "pod-c", model)
    r = run(["bash", "brady-gpt/pod/run_context_test.sh"], script(home, MODEL_SHA="0" * 64), "a wrong checksum stops the run before anything else", expect=1)
    check("and scores nothing", not (home / "brady-gpt/data/generations/7b").exists())

    print("== the background launch, exactly as NEXT_SESSION.md gives it")
    home = new_pod(sy, "pod-d", model)
    e = script(home, FAKE_POD_HOME=str(home), LIMIT="6", HF_TOKEN="")
    launch = "setsid nohup bash ~/brady-gpt/pod/run_context_test.sh </dev/null >>~/context-test.log 2>&1 &"
    start = time.time()
    # Held by pipes, as real ssh holds its channel: this waits if anything keeps them open.
    subprocess.run(["ssh", "-p", "22", "user@pod", launch], env=e, check=True, timeout=30, capture_output=True, stdin=subprocess.DEVNULL)
    check("the launch returns at once, with nothing holding ssh open", time.time() - start < 10, f"{time.time() - start:.1f}s")
    assert_old = subprocess.run(["ssh", "-p", "22", "user@pod", "cd ~ && sleep 4 </dev/null >/dev/null 2>&1 &"], env=e, capture_output=True, stdin=subprocess.DEVNULL)
    check("(control: the plan's first version, cd && ... &, does hold it open)", time.time() - start >= 4 and assert_old.returncode == 0, f"{time.time() - start:.1f}s")
    log_file, seen = home / "context-test.log", ""
    while time.time() - start < 300 and "context test done" not in seen and "FAILED" not in seen:
        time.sleep(3)
        seen = log_file.read_text() if log_file.exists() else ""
    check("it keeps running after ssh has gone, and finishes", "context test done" in seen, seen[-200:] if "context test done" not in seen else "")

    print("== pull.sh")
    mac = sy / "mac"
    (mac / "pod").mkdir(parents=True)
    shutil.copy(HERE / "pod/pull.sh", mac / "pod/pull.sh")
    e2 = {**e, "_CWD": str(mac), "HOME": str(Path.home())}
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], e2, "a pod with no checkpoints still pulls")
    check("the results arrived", (mac / "data/generations/7b/quick/context-test.json").exists())
    check("it says wiping is fine", "done. Then" in r.stdout)
    pod_out = home / "brady-gpt/data/generations/7b"
    bad = sy / "bin-bad"
    bad.mkdir()
    (bad / "rsync").write_text("#!/bin/sh\nexit 1\n")
    (bad / "rsync").chmod(0o755)
    no_copy = {**e2, "PATH": f"{bad}:{e2['PATH']}"}
    (pod_out / "new-result.json").write_text("{}")
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], no_copy, "a new file that didn't copy stops pull.sh", expect=1)
    check("and warns against wiping", "do NOT wipe" in r.stdout)
    (pod_out / "new-result.json").unlink()
    (pod_out / "quick/context-test.json").write_text('{"answers": 618}')
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], no_copy, "an old file of the same name on the Mac stops pull.sh", expect=1)
    check("and warns against wiping", "do NOT wipe" in r.stdout)
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], e2, "a working copy fixes that")
    down = sy / "bin-down"
    down.mkdir()
    (down / "ssh").write_text("#!/bin/sh\nexit 255\n")
    (down / "ssh").chmod(0o755)
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], {**e2, "PATH": f"{down}:{e2['PATH']}"}, "the network being down stops pull.sh", expect=255)
    check("and it doesn't say wiping is fine", "done. Then" not in r.stdout)
    (pod_out / "context-test.running").write_text("")
    r = run(["bash", "pod/pull.sh", "user@pod", "22"], e2, "an unfinished run stops pull.sh", expect=1)
    check("and says the test hasn't finished", "still running or failed" in r.stdout)
    (pod_out / "context-test.running").unlink()
    ckpt = home / "brady-gpt/checkpoints/7b"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter.bin").write_text("x")
    run(["bash", "pod/pull.sh", "user@pod", "22"], e2, "a pod with checkpoints still pulls them")
    check("and they arrive", (mac / "checkpoints/7b/adapter.bin").exists())


def main():
    sy = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    sy.mkdir(parents=True, exist_ok=True)
    unit_tests()
    print("== building fake data and a tiny OLMo 3")
    fake_data(sy / "data")
    tiny_model(sy / "tiny")
    env = {**os.environ, "VLLM_PYTHON": sys.executable, "PYTHONPATH": str(HERE / "pod/fake_vllm"), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    end_to_end(sy, sy / "tiny", env)
    pod_scripts(sy, sy / "tiny", env)
    print(f"\n{'ALL PASSED' if not FAILED else str(len(FAILED)) + ' FAILED: ' + '; '.join(FAILED)}  ({sy})")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
