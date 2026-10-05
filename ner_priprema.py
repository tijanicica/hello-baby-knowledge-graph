"""
Faza 2.2 - Priprema podataka za NER (BERTić).

Koraci:
    1. minimalno čišćenje teksta (Unicode NFC, razmaci) - bez izbacivanja stop reči,
       interpunkcije i velikih slova, jer transformer modelu treba ceo kontekst
    2. tokenizacija i pretvaranje anotacija [tekst|TIP] u BIO oznake
    3. podela na trening / validacioni / test skup PO DIJALOZIMA
       (+ 5 foldova za unakrsnu validaciju)

Izlaz (folder data_ner/, ne ide na GitHub jer sadrži tekst dijaloga):
    train.jsonl, dev.jsonl, test.jsonl   - jedna replika po redu: tokens + ner_tags
    labels.json                          - spisak BIO oznaka (O, B-..., I-...)
    split.json                           - koji dijalozi su u kom skupu, i 5 foldova
"""
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "osnovne_anotacije"
OUT = BASE / "data_ner"
OUT.mkdir(exist_ok=True)

SEED = 42
N_DEV, N_TEST = 15, 30   # ostatak (105) ide u trening
N_FOLDS = 5

SPAN_RE = re.compile(r"\[([^\[\]|]+)\|\s*([A-Z_]+)\s*\]") #ovo prepoznaje [tekst|TIP].
# brojevi sa decimalom (37.8), reči sa crticom (dva-tri), pa pojedinačni znaci interpunkcije
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|\w+(?:-\w+)*|[^\w\s]") # ovo deli tekst na odvojene reci


def clean(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip() #sve svodi na jedan razmak i uklanja razmake sa pocetka i kraja


def parse(annotated):
    """Vraća čist tekst i listu entiteta (početak, kraj, tip) po pozicijama karaktera.
    Na primer funkcija vraća:
"Bebici se javilo neko crvenilo po vratu pa sad ne znamo zbog čega ni šta da radimo."
[(17, 39, "SYMPTOM"), (47, 55, "UNCERTAINTY")]"""
    plain, entities, pos = [], [], 0
    for m in SPAN_RE.finditer(annotated):
        plain.append(annotated[pos:m.start()])
        start = sum(len(p) for p in plain)
        span = m.group(1)
        # razmaci na ivicama raspona ne pripadaju entitetu
        lead = len(span) - len(span.lstrip())
        plain.append(span)
        entities.append((start + lead, start + len(span.rstrip()), m.group(2)))
        pos = m.end()
    plain.append(annotated[pos:])
    return "".join(plain), entities


def to_bio(text, entities):
    """Tokenizuje tekst i dodeljuje BIO oznaku svakom tokenu."""
    misaligned = 0
    spans = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)] # Lista svih tokena sa pozicijama: ("Bebici", 0, 6), ("se", 7, 9), … ("neko", 17, 21), ("crvenilo", 22, 30),
    labels = ["O"] * len(spans)
    for e_start, e_end, label in entities:
        inside = [i for i, (_, s, e) in enumerate(spans) if s < e_end and e > e_start]
        if not inside:
            misaligned += 1
            continue
        if spans[inside[0]][1] != e_start or spans[inside[-1]][2] != e_end:
            misaligned += 1  # granica entiteta pada usred tokena; zadržavamo cele tokene
        for k, i in enumerate(inside):
            labels[i] = ("B-" if k == 0 else "I-") + label
    tokens = [t for t, _, _ in spans]
    return tokens, labels, misaligned


# ---------------------------------------------------------------------------
# 1-2. Učitavanje, čišćenje, BIO
# ---------------------------------------------------------------------------
records, n_entities, n_misaligned = [], 0, 0
for path in sorted(DATA.glob("*.json"), key=lambda p: int(p.stem)):
    with open(path, encoding="utf-8") as f:
        turns = json.load(f)["annotations"]
    for i, turn in enumerate(turns):
        text, entities = parse(clean(turn["annotated_text"]))
        tokens, tags, bad = to_bio(text, entities)
        n_entities += len(entities)
        n_misaligned += bad
        records.append({
            "dialog_id": int(path.stem),
            "turn_idx": i,
            "speaker": turn["speaker"],
            "tokens": tokens,
            "ner_tags": tags,
        })

entity_types = sorted({t[2:] for r in records for t in r["ner_tags"] if t != "O"})
label_list = ["O"] + [f"{p}-{t}" for t in entity_types for p in ("B", "I")]

print(f"Replika: {len(records)}, tokena: {sum(len(r['tokens']) for r in records)}")
print(f"Entiteta: {n_entities}, tipova: {len(entity_types)}, BIO oznaka: {len(label_list)}")
print(f"Entiteta čija granica ne pada na granicu tokena: {n_misaligned}")
print(f"Najduža replika: {max(len(r['tokens']) for r in records)} tokena")

# ---------------------------------------------------------------------------
# 3. Podela po dijalozima
# ---------------------------------------------------------------------------
dialog_ids = sorted({r["dialog_id"] for r in records})
rng = random.Random(SEED)
shuffled = dialog_ids[:]
rng.shuffle(shuffled)
test_ids = sorted(shuffled[:N_TEST])
dev_ids = sorted(shuffled[N_TEST:N_TEST + N_DEV])
train_ids = sorted(shuffled[N_TEST + N_DEV:])
folds = [sorted(shuffled[k::N_FOLDS]) for k in range(N_FOLDS)]

splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
for name, ids in splits.items():
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if r["dialog_id"] in ids:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

(OUT / "labels.json").write_text(json.dumps(label_list, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "split.json").write_text(json.dumps({**splits, "seed": SEED, "folds": folds}, indent=1), encoding="utf-8")

# Provera: da li su retki tipovi zastupljeni u svakom skupu
counts = {name: Counter(t[2:] for r in records if r["dialog_id"] in ids
                        for t in r["ner_tags"] if t.startswith("B-"))
          for name, ids in splits.items()}
print(f"\nDijaloga: train {len(train_ids)}, dev {len(dev_ids)}, test {len(test_ids)}")
print(f"\n{'tip':<20}{'train':>7}{'dev':>7}{'test':>7}")
for t in sorted(entity_types, key=lambda t: -counts["train"][t]):
    flag = "  <- malo primera u testu" if counts["test"][t] < 5 else ""
    print(f"{t:<20}{counts['train'][t]:>7}{counts['dev'][t]:>7}{counts['test'][t]:>7}{flag}")

example = next(r for r in records if r["dialog_id"] == 100 and r["turn_idx"] == 2)
print("\nPrimer (dijalog 100, replika 3):")
for tok, tag in zip(example["tokens"], example["ner_tags"]):
    print(f"  {tok:<12}{tag}")
print(f"\nSačuvano u: {OUT}")
