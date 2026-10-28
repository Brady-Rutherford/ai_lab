"""Every Minerva name I have, for masking text and for checking that the masking worked.

Built from the CS156 privacy scanner's own needles (`oncue.privacy`): every roster in
scribe/raw, my own names excluded. prepare_data.py masks with it and privacy_check.py gates
with it, so the two can never disagree about what counts as a name.

Each class's RosterMasker only knows that class's roster, so it misses classmates from other
classes; this pass catches them. Names that are also common English words are the hard
case, and they come in two kinds, told apart by how my transcripts use them (Deepgram
capitalises proper names):
- name-like (Daniel, say): mostly capitalised mid-sentence. Masked everywhere.
- word-like (Will, May): mostly lower case. At a sentence start they are masked only when
  a name signal follows (see NAME_SIGNAL); otherwise they are left alone and
  privacy_check.py reports them as ambiguous.
Short forms of roster first names (the first few letters of one) are not on any roster, so the
scanner can't see them; they are masked as capitalised prefixes of 3+ letters of a roster first
name that are not English words (not before "'t", so "Hasn't" survives). And a capitalised word
right before a comma or question mark (a vocative: "Shinn, what do you think?") within one edit
of a roster name is masked too: speech-to-text misspells names, and the CS156 masker only
fuzzy-matches mid-sentence.
"""

import json
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

SRC = Path(os.environ.get("CS156_DIR", "~/cs156-ml/pipeline1/data")).expanduser()
sys.path.insert(0, str(SRC.parent))
from oncue import config, privacy  # noqa: E402
from scribe_core.text import levenshtein  # noqa: E402

NAME = "[NAME]"
# after a sentence end, a line break, or a "Speaker:" label
SENTENCE_START = re.compile(r"(?:^|[.?!][\"')\]”’]*\s+|\n\s*|:\s+)$")
# When in doubt, mask: a few masked ordinary words cost little, a leaked name a lot. So a
# name signal is: a possessive; punctuation right after it ("Will,", "Will?", "Will." as a
# whole sentence, "Al-"); a verb or adverb a person takes ("Will is", "Will took", "Will also");
# another name after it ("Will [NAME]", "Will and [NAME]"); or a capitalised word (a surname).
# Signals stay on the name's own line ([ \t], not \s), so the masker (one turn at a time) and
# privacy_check.py (whole prompts, one turn per line) always agree.
NAME_SIGNAL = re.compile(r"(?:['’]s\b|[ \t]*[,?.!-]|[ \t]+(?:is|was|has|had|does|did|doesn't|didn't"
                         r"|isn't|wasn't|looks|seems|wants|said|says|mentioned|brought|pointed|asked"
                         r"|raised|made|talked|agrees|agreed|thinks|thought|took|loves|likes|knows|comes|goes"
                         r"|gets|needs|takes|gives|tells"
                         r"|told|wrote|also|just|really|actually|and I)\b"
                         r"|[ \t]+(?:and[ \t]+)?\[NAME\]|[ \t]+[A-Z][a-z]+)")
EDGE = ".,!?;:\"'()[]“”‘’"


@lru_cache(maxsize=1)
def needles():
    config.load_env()  # the ID salt, so salted ids are checked too
    try:
        salt = config.id_salt()
    except RuntimeError:
        salt = None
    roster = privacy.load_roster(config.RAW_DIR)
    ids = privacy.load_hashed_ids(config.TABLES_DIR)
    return privacy._needles(roster, ids, salt=salt, extra_terms=[],
                            exclude=list(config.BRADY_VARIANTS), allow=[])


@lru_cache(maxsize=1)
def nickname_re():
    """Capitalised short forms of roster names: prefixes of 3+ letters that are not common
    words, not themselves full names (those are in needles()), and not prefixes of mine."""
    common = privacy._common_words()
    mine = {v.lower() for v in config.BRADY_VARIANTS}
    names = {t.lower() for t in privacy.roster_terms(privacy.load_roster(config.RAW_DIR))
             if " " not in t and t.isalpha()}
    prefixes = {n[:k] for n in names - mine for k in range(3, len(n))}
    prefixes = {p for p in prefixes if p not in common and p not in names
                and not any(m.startswith(p) for m in mine)}
    alts = sorted((re.escape(p.capitalize()) for p in prefixes), key=len, reverse=True)
    return re.compile(r"(?<![\w\[])(?:" + "|".join(alts) + r")(?!\w|['’]t\b)") if alts else None


VOCATIVE = re.compile(r"(?<![\w\[])([A-Z][a-z]{3,})(?=[ \t]*[,?])")


@lru_cache(maxsize=1)
def _names_by_length():
    common = privacy._common_words()
    by_len = {}
    for t in privacy.roster_terms(privacy.load_roster(config.RAW_DIR)):
        low = t.lower()
        if " " not in t and low.isalpha() and len(low) >= config.MASK_LEV_MIN_LEN and low not in common:
            by_len.setdefault(len(low), set()).add(low)
    return by_len


def is_misspelled_name(word):
    """Within one edit of a roster name (the CS156 masker's own fuzzy rule), not a common word."""
    low = word.lower()
    if low in privacy._common_words() or low in {v.lower() for v in config.BRADY_VARIANTS}:
        return False
    by_len = _names_by_length()
    return any(levenshtein(low, n) <= config.MASK_LEV_MAX
               for k in (len(low) - 1, len(low), len(low) + 1) for n in by_len.get(k, ()))


def is_common_word_name(term):
    nd = needles()
    return bool(nd.names_cs and nd.names_cs.fullmatch(term)
                and not (nd.names_ci and nd.names_ci.fullmatch(term)))


@lru_cache(maxsize=1)
def name_like():
    """Common-word roster names that my transcripts mostly use as names: capitalised
    mid-sentence at least 3 times and at least as often as in lower case."""
    common = privacy._common_words()
    mine = {v.lower() for v in config.BRADY_VARIANTS}
    candidates = {t.lower() for t in privacy.roster_terms(privacy.load_roster(config.RAW_DIR))
                  if " " not in t and t.lower() in common and t.lower() not in mine}
    lower, capital = Counter(), Counter()
    for path in SRC.glob("transcripts/*/*_ids.jsonl"):
        for line in open(path):
            if not line.strip():
                continue
            prev = None
            for tok in (json.loads(line).get("text_clean") or "").split():
                core = tok.strip(EDGE)
                low = core.lower()
                if low in candidates:
                    if core.islower():
                        lower[low] += 1
                    elif core[:1].isupper() and prev and not prev.endswith((".", "?", "!")):
                        capital[low] += 1
                prev = tok
    return frozenset(w for w in candidates if capital[w] >= 3 and capital[w] >= lower[w])


def is_ambiguous(text, start, end):
    """A word-like common-word name at a sentence start with no name signal after it
    ("Will we...", "May I...")."""
    if text[start:end].lower() in name_like():
        return False
    return bool(SENTENCE_START.search(text[max(0, start - 6):start])) \
        and not NAME_SIGNAL.match(text, end)


def mask_names(text, header=False):
    """[NAME] for every roster name. `header` text (course titles, LO descriptions) is
    institutional, so common-word names there are left alone."""
    nd = needles()
    if nd.names_ci:
        text = nd.names_ci.sub(NAME, text)
    if not header and nickname_re():
        text = nickname_re().sub(NAME, text)
    if not header:
        text = VOCATIVE.sub(lambda m: NAME if is_misspelled_name(m.group(1)) else m.group(0), text)
    if nd.names_cs and not header:
        src = text
        text = nd.names_cs.sub(
            lambda m: m.group(0) if is_ambiguous(src, m.start(), m.end()) else NAME, src)
    return text
