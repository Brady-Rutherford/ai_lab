"""Build the post-training data that teaches the model to answer like me.

Class data is used for post-training only; pretraining stays on general web text. Each
example pairs an ask (a professor's question in class, a pre-class work prompt, a poll) with
what I actually said or wrote, plus the LOs/HCs that class focused on.

Run from this folder, inside the CS156 environment so its roster masker and Forum clock are
reused:

    uv run --project ~/cs156-ml python prepare_data.py                         # full build
    uv run --project ~/cs156-ml python prepare_data.py --limit 2 --out <dir>   # smoke test

Reads (from $CS156_DIR, default ~/cs156-ml/pipeline1/data):
    transcripts/{train,validation,test}/*_ids.{json,jsonl}   diarized, calibrated class speech
    scribe/raw/{cid}/class.json                               rosters, for name masking
    scribe/raw/{cid}/focused_outcomes.json                    the LOs/HCs each class focused on
    scribe/raw/{cid}/outcome_assessments.json                 professors' grades on my work
    scribe/raw/{cid}/{poll_responses,events}.json             to attach those grades
    tables/pcw_blocks.parquet, tables/polls.parquet           my written answers
    03_split/split.csv                                        the class-level split

Writes (under --out, default data/, gitignored):
    sft/{train,val,test}.jsonl              my examples, TRL prompt-completion format
    sft_classmates/{train,val,test}.jsonl   classmates' class answers, variant B only
    transcripts/{train,val,test}.parquet    one masked document per class. Not for training:
                                            for the tokenizer study and bits-per-byte evals
    stats.json

Each row: {"prompt": [user message], "completion": [assistant message], "source", "speaker",
"class_id", "los", "grade", "grade_z"}. `grade` is the mean professor score on that answer
(polls and spoken turns only); `grade_z` normalizes it per grader, since some grade harder.
Grader ids are used in memory only and never written.

Privacy: names go through the CS156 roster masker (classmates -> [NAME]); classmates'
answers that disclose personal details (family, health, where they are from) are dropped.
The split is by class, so a held-out class never leaks into training through any source.
Classes with written work but no transcript have no split; they go to train.
"""

import argparse
import bisect
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import DATA, ME, TAG_CLASSMATE, TAG_ME, write_jsonl
from names import mask_names

SRC = Path(os.environ.get("CS156_DIR", "~/cs156-ml/pipeline1/data")).expanduser()
sys.path.insert(0, str(SRC.parent))  # the pipeline1 folder, home of the oncue package
from oncue.config import calibration_qa  # noqa: E402
from oncue.events import (  # noqa: E402
    RosterMasker, kebab, parse_voice, roster_from_class, video_clock)

SPLITS = {"train": "train", "validation": "val", "test": "test"}
MIN_TURN_WORDS = 8      # shorter turns are "yeah", "can you hear me", ...
LOOKBACK_TURNS = 6      # how far back from a turn to look for the professor's ask
ASK_WORDS = 150         # keep the end of the professor's turn, where the question and call-on sit
BETWEEN_WORDS = 200     # replies between the ask and the answer, newest kept
MAX_ANSWER_WORDS = 1200
MAX_LOS = 6             # focused LOs/HCs listed in a prompt
MATCH_SLACK_S = 3.0     # a graded voice event may sit this far outside my utterance times
MIN_GRADER_STD = 0.5    # floor for per-grader spread, so one-grade graders don't explode
DISCLOSURE = re.compile(
    r"\b(my (mom|mother|dad|father|parents?|family|brother|sister|grand\w+|aunt|uncle|cousin"
    r"|boyfriend|girlfriend|partner|wife|husband|son|daughter|home ?town|home country|country"
    r"|village|religion|church|mosque|temple|health|doctor|therapist|diagnosis|medication"
    r"|disability|culture)|i grew up|growing up in|where i'?m from|i'?m from|back home)\b",
    re.IGNORECASE)

ME_ID = json.load(open(SRC / "scribe/me.json"))["id"]
SPLIT = pd.read_csv(SRC / "03_split/split.csv").set_index("class_id")
RESIDUALS = pd.read_parquet(SRC / "tables/classes.parquet",
                            columns=["class_id", "resid_med_s", "resid_p95_s"]).set_index("class_id")


def raw(cid, name):
    path = SRC / f"scribe/raw/{cid}/{name}.json"
    return json.load(open(path)) if path.exists() else None


def focused_los(cid):
    """[(hashtag, description)] for the LOs/HCs a class focused on, in Forum's order."""
    seen = {}
    for item in raw(cid, "focused_outcomes") or []:
        lo = item.get("learning-outcome") or {}
        tag = lo.get("hashtag")
        if tag and tag not in seen:
            seen[tag] = mask_names(" ".join((lo.get("description") or "").split()), header=True)
    return list(seen.items())[:MAX_LOS]


def class_info(cid):
    """Course, session title, name masker and focused LOs/HCs for one class."""
    klass = kebab(raw(cid, "class") or {})
    roster = roster_from_class(klass, ME_ID)
    if cid in SPLIT.index:
        row = SPLIT.loc[cid]
        course, title = f"{row.course_code} {row.course_title}", row.title
    else:
        c = (klass.get("section") or {}).get("course") or {}
        course = " ".join(str(c.get(k)) for k in ("code", "title") if c.get(k)) or "Unknown course"
        title = klass.get("title") or f"class {cid}"
    title = mask_names(str(title), header=True)
    return {"course": course, "title": title, "los": focused_los(cid), "klass": klass,
            "masker": RosterMasker(roster.names, roster.me_names)}


def mask(masker, text, header=False):
    """Mask names: the class's own roster first (with its fuzzy matching), then every roster
    (names.py). Line by line, so paragraph breaks in written answers survive."""
    lines = (masker.mask_text(line) or "" for line in str(text or "").split("\n"))
    return mask_names("\n".join(lines).replace("[ME]", ME).strip(), header=header)


def split_of(cid):
    return SPLITS[SPLIT.loc[cid, "split"]] if cid in SPLIT.index else "train"


def header(tag, kind, info, segment=None):
    lines = [tag, f"{kind} in {info['course']} - {info['title']}"]
    if segment:
        lines.append(f"Activity: {segment}")
    if info["los"]:
        lines.append("Focused LOs/HCs:")
        lines += [f"- {t}: {d}" if d else f"- {t}" for t, d in info["los"]]
    return "\n".join(lines)


def example(user, answer, source, speaker, cid, info, grades=()):
    return {"prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": answer}],
            "source": source, "speaker": speaker, "class_id": int(cid),
            "los": [t for t, _ in info["los"]], "_grades": list(grades)}


# ----------------------------------------------------------------------------- transcripts


def speaker_turns(lines, masker, timeline):
    """Merge consecutive utterances by one speaker into turns, with display labels and times,
    then mask each turn."""
    starts = [seg["t"] for seg in timeline]
    labels, turns = {}, []
    for u in lines:
        if u["is_me"]:
            who = ME
        elif u["speaker_role"] == "professor":
            who = "Professor"
        elif u["speaker_role"] == "student":
            who = labels.setdefault(u["speaker_id"], f"Student {len(labels) + 1}")
        else:
            who = "Speaker"
        text = (u.get("text_clean") or u.get("text") or "").strip()
        if not text:
            continue
        i = bisect.bisect_right(starts, u["start"]) - 1
        segment = mask(masker, timeline[i]["title"]) if i >= 0 else None
        if turns and turns[-1]["who"] == who and turns[-1]["segment"] == segment:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = u["end"]
        else:
            turns.append({"who": who, "text": text, "segment": segment,
                          "start": u["start"], "end": u["end"]})
    # Mask whole turns, not utterances: the masker treats a turn's first word as a sentence
    # start, so masking each utterance would skip names at every utterance boundary.
    for t in turns:
        t["text"] = mask(masker, t["text"])
    return [t for t in turns if t["text"]]


def render_class(info, date, turns):
    out, segment = [f"{info['course']}\n{info['title']} ({date})\n"], object()
    for t in turns:
        if t["segment"] != segment:
            segment = t["segment"]
            if segment:
                out.append(f"\n[{segment}]")
        out.append(f"{t['who']}: {t['text']}")
    return "\n".join(out) + "\n"


def find_ask(turns, i):
    """Index of the professor turn that turn i answers: the latest question within the
    lookback, else the latest professor turn at all, else None (a student-only stretch)."""
    window = range(i - 1, max(-1, i - 1 - LOOKBACK_TURNS), -1)
    profs = [j for j in window if turns[j]["who"] == "Professor"]
    if not profs:
        return None
    return next((j for j in profs if "?" in turns[j]["text"]), profs[0])


def answer_examples(cid, info, turns, speaker, grades_by_turn, dropped):
    """One example per substantial answer by `speaker` ("brady" or "classmate"): the
    professor's ask, plus the replies since, -> what was said."""
    tag = TAG_ME if speaker == "brady" else TAG_CLASSMATE
    for i, t in enumerate(turns):
        mine = t["who"] == ME
        if mine != (speaker == "brady") or not (mine or t["who"].startswith("Student ")):
            continue
        if len(t["text"].split()) < MIN_TURN_WORDS:
            continue
        j = find_ask(turns, i)
        if j is None:
            continue
        if not mine and DISCLOSURE.search(t["text"]):
            dropped["personal_disclosure"] += 1
            continue
        ask = " ".join(turns[j]["text"].split()[-ASK_WORDS:])
        between, budget = [], BETWEEN_WORDS
        for c in reversed(turns[j + 1:i]):
            budget -= len(c["text"].split())
            if budget < 0:
                break
            between.insert(0, f"{c['who']}: {c['text']}")
        user = "\n".join([header(tag, "Class discussion", info, t["segment"]), "",
                          f"Professor: {ask}", *between, "", "What do you say?"])
        yield example(user, t["text"], "class_turn", speaker, cid, info, grades_by_turn.get(i, ()))


def spoken_grades(cid, meta, info, turns, assessments, counts):
    """{turn index: [(score, grader)]}: each grade on a voice event of mine, attached to the
    turn of mine it overlaps most, via the CS156 pipeline's calibrated Forum clock. Only in
    classes that pass the pipeline's own clock QA (`oncue.config.calibration_qa`)."""
    graded = defaultdict(list)
    for a in assessments:
        if a.get("recording-session-event-id") is not None and a.get("score") is not None:
            graded[str(a["recording-session-event-id"])].append((a["score"], a.get("grader-user-id")))
    if not graded:
        return {}
    cal = meta["alignment"]["calibration"]
    resid = RESIDUALS.loc[cid] if cid in RESIDUALS.index else {}
    ok, _ = calibration_qa(cal, resid.get("resid_med_s"), resid.get("resid_p95_s"))
    if not ok:
        counts["spoken_clock_failed_qa"] += sum(map(len, graded.values()))
        return {}
    try:
        clock = video_clock(None, cal, info["klass"])
        voice = parse_voice(raw(cid, "events") or [], clock, meta["alignment"]["main_room_id"])
    except Exception:
        counts["spoken_clock_failed"] += sum(map(len, graded.values()))
        return {}
    mine = [i for i, t in enumerate(turns) if t["who"] == ME]
    out = defaultdict(list)
    for ev in voice.itertuples():
        scores = graded.get(str(ev.event_id))
        if not scores:
            continue
        overlap = {i: min(ev.end_s, turns[i]["end"] + MATCH_SLACK_S)
                   - max(ev.start_s, turns[i]["start"] - MATCH_SLACK_S) for i in mine}
        best = max(overlap, key=overlap.get, default=None)
        if best is None or overlap[best] < 0:
            counts["spoken_unmatched"] += len(scores)
            continue
        out[best] += scores
        counts["spoken_matched"] += len(scores)
    return out


# ----------------------------------------------------------------------------- written work


def text_key(text):
    return " ".join(re.sub(r"<[^>]+>", " ", str(text or "")).lower().split())[:200]


def poll_grades(cid, assessments, masker):
    """{answer text key: [(score, grader)]} for my graded poll answers in one class. The key is
    taken after masking, because tables/polls.parquet stores my answers already masked."""
    by_session = defaultdict(list)
    for a in assessments:
        if a.get("poll-session-id") is not None and a.get("score") is not None:
            by_session[str(a["poll-session-id"])].append((a["score"], a.get("grader-user-id")))
    out = defaultdict(list)
    for r in raw(cid, "poll_responses") or []:
        if str(r.get("user-id")) == str(ME_ID) and str(r.get("poll-session-id")) in by_session:
            out[text_key(masker.mask_text(r.get("response")))] += by_session[str(r["poll-session-id"])]
    return out


def normalize_grades(rows):
    """Turn each row's in-memory (score, grader) list into `grade` and per-grader `grade_z`."""
    by_grader = defaultdict(list)
    for row in rows:
        for score, grader in row["_grades"]:
            by_grader[grader].append(score)
    stats = {g: (statistics.mean(s), max(statistics.pstdev(s), MIN_GRADER_STD))
             for g, s in by_grader.items()}
    for row in rows:
        grades = row.pop("_grades")
        if grades:
            row["grade"] = round(statistics.mean(s for s, _ in grades), 3)
            row["grade_z"] = round(statistics.mean((s - stats[g][0]) / stats[g][1]
                                                   for s, g in grades), 3)
        else:
            row["grade"] = row["grade_z"] = None


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="transcribed classes per split (smoke tests)")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()

    docs, mine, classmates = defaultdict(list), defaultdict(list), defaultdict(list)
    counts, info, poll_keys = Counter(), {}, {}

    def info_for(cid):
        if cid not in info:
            info[cid] = class_info(cid)
        return info[cid]

    for folder, split in SPLITS.items():
        paths = sorted((SRC / "transcripts" / folder).glob("*_ids.jsonl"))[:args.limit]
        for path in paths:
            meta = json.load(open(str(path).removesuffix("l")))
            cid = meta["class"]["id"]
            ci = info_for(cid)
            lines = [json.loads(x) for x in open(path) if x.strip()]
            turns = speaker_turns(lines, ci["masker"], meta.get("timeline") or [])
            assessments = raw(cid, "outcome_assessments") or []
            graded = spoken_grades(cid, meta, ci, turns, assessments, counts)
            poll_keys[cid] = poll_grades(cid, assessments, ci["masker"])
            docs[split].append(render_class(ci, meta["class"]["date"], turns))
            mine[split].extend(answer_examples(cid, ci, turns, "brady", graded, counts))
            classmates[split].extend(answer_examples(cid, ci, turns, "classmate", {}, counts))
    keep = set(info) if args.limit else None  # a smoke test only touches the classes it read

    pcw = pd.read_parquet(SRC / "tables/pcw_blocks.parquet")
    pcw = pcw[(pcw.answer_words >= 5) & (pcw.answer_words <= MAX_ANSWER_WORDS)
              & (pcw.prompt_text.str.len() > 20)]
    for r in pcw.itertuples():
        if keep is not None and r.class_id not in keep:
            continue
        ci = info_for(r.class_id)
        user = f"{header(TAG_ME, 'Pre-class work', ci)}\n\n{mask(ci['masker'], r.prompt_text)}"
        mine[split_of(r.class_id)].append(
            example(user, mask(ci["masker"], r.answer_text), "pcw", "brady", r.class_id, ci))

    polls = pd.read_parquet(SRC / "tables/polls.parquet")
    for r in polls[polls.my_answer_text.notna()].itertuples():
        if keep is not None and r.class_id not in keep:
            continue
        if not 3 <= len(r.my_answer_text.split()) <= MAX_ANSWER_WORDS:
            continue
        ci = info_for(r.class_id)
        if r.class_id not in poll_keys:
            poll_keys[r.class_id] = poll_grades(r.class_id, raw(r.class_id, "outcome_assessments") or [],
                                              ci["masker"])
        grades = poll_keys[r.class_id].get(text_key(r.my_answer_text), [])
        counts["poll_graded" if grades else "poll_ungraded"] += 1
        user = f"{header(TAG_ME, 'In-class poll', ci)}\n\n{mask(ci['masker'], r.prompt_masked)}"
        mine[split_of(r.class_id)].append(
            example(user, mask(ci["masker"], r.my_answer_text), "poll", "brady", r.class_id, ci, grades))

    normalize_grades([row for rows in mine.values() for row in rows])
    for rows in classmates.values():
        for row in rows:
            row.pop("_grades")
            row["grade"] = row["grade_z"] = None

    stats = {"counts": dict(counts), "sft": {}, "sft_classmates": {}, "transcripts": {}}
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)
    for split in SPLITS.values():
        write_jsonl(args.out / f"sft/{split}.jsonl", mine[split])
        write_jsonl(args.out / f"sft_classmates/{split}.jsonl", classmates[split])
        pd.DataFrame({"text": docs[split]}).to_parquet(args.out / f"transcripts/{split}.parquet")
        stats["transcripts"][split] = {"classes": len(docs[split])}
        for name, rows in (("sft", mine[split]), ("sft_classmates", classmates[split])):
            by_source = defaultdict(lambda: {"examples": 0, "graded": 0, "answer_words": 0})
            for row in rows:
                s = by_source[row["source"]]
                s["examples"] += 1
                s["graded"] += row["grade"] is not None
                s["answer_words"] += len(row["completion"][0]["content"].split())
            stats[name][split] = dict(by_source)
    json.dump(stats, open(args.out / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
