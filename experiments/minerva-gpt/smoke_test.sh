#!/bin/sh
# Every pipeline step once, on fake data from synthetic.py. Checks the code runs end to end;
# reads no class data and writes nothing into data/ or checkpoints/.
#
#     sh smoke_test.sh [dir]                     # dir defaults to a fresh temp folder
#     GEN_ENGINE=vllm sh smoke_test.sh           # on the GPU pod: sample with vLLM too
set -e
cd "$(dirname "$0")"
SY=${1:-$(mktemp -d)}
ENGINE=${GEN_ENGINE:-hf}
mkdir -p "$SY"
# A failing step stops the test with its own exit code (piping into grep would hide it).
run() {
  echo "== $*"
  if uv run python "$@" > "$SY/step.log" 2>&1; then status=0; else status=$?; fi
  tr '\r' '\n' < "$SY/step.log" | grep -v -E "Redirects|it/s\]|s/it\]|^\s*$" | tail -6
  return $status
}

run synthetic.py "$SY"
run train_sft.py --variant A --data "$SY" --out "$SY/sft-A" --max-steps 20 --effective-batch 8
run train_sft.py --variant B --data "$SY" --out "$SY/sft-B" --max-steps 10 --effective-batch 8
run generate.py --adapter "$SY/sft-A" --variant A --data "$SY" --n 2 --limit 2 --engine "$ENGINE" --out "$SY/gen-A.jsonl"
run generate.py --adapter "$SY/sft-B" --variant B --data "$SY" --n 2 --limit 2 --retrieve --engine "$ENGINE" --out "$SY/gen-B.jsonl"
run voice_score.py --adapter "$SY/sft-B" --variant B --data "$SY" --validate val --n 4 --score "$SY/gen-B.jsonl"
run voice_score.py --adapter "$SY/sft-A" --variant A --data "$SY" --score "$SY/gen-A.jsonl"
run reward_model.py --data "$SY" --train --model "$SY/grade.pkl"
run reward_model.py --data "$SY" --score "$SY/gen-B.voice.jsonl" --model "$SY/grade.pkl"
run eval_regurgitation.py --data "$SY" "$SY/gen-A.jsonl" "$SY/gen-B.jsonl" \
  || echo "   (a hit is possible here: the fake data repeats the same few sentences)"

# ranking page: serve it, fetch one ask, save one ranking, stop
uv run python rank_server.py --a "$SY/gen-A.jsonl" --b "$SY/gen-B.jsonl" --data "$SY" --port 8799 >/dev/null 2>&1 &
server=$!
sleep 8
ask=$(curl -s "http://127.0.0.1:8799/api/next?mode=rank")
id=$(printf '%s' "$ask" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
n=$(printf '%s' "$ask" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['answers']))")
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"id\": \"$id\", \"voice\": $(python3 -c "import json; print(json.dumps([i % 2 == 0 for i in range($n)]))"), \"best\": 0}" \
  "http://127.0.0.1:8799/api/rank"
echo
kill $server
echo "== rankings saved: $(wc -l < "$SY/rankings.jsonl")"

run train_dpo.py --variant B --rankings "$SY/rankings.jsonl" --auto "$SY/gen-B.voice.grade.jsonl" \
  --sft "$SY/sft-B" --out "$SY/dpo-B" --grade-model "$SY/grade.pkl" --max-steps 2
echo "== smoke test passed: $SY"
