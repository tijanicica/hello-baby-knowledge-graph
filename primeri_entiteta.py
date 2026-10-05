"""
Primeri upotrebe za svaki od 30 tipova entiteta (pomoć pri definisanju ontologije).

Za svaki tip bira najčešći oblik entiteta u bazi i prikazuje jedan isečak
razgovora u kom se on javlja: prethodnu repliku, repliku sa entitetom i
sledeću repliku. Entitet je označen kao »tekst«, a ostale oznake su uklonjene
da bi se razgovor video onako kako je izgovoren.

Ispis ide na ekran i u rezultati_faza1/primeri_entiteta.txt.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "rezultati_faza1"
OUT.mkdir(exist_ok=True)

N_PRIMERA = 1          # koliko primera po tipu entiteta
KONTEKST = 1           # koliko replika pre i posle prikazati
MIN_RECI, MAX_RECI = 6, 40  # biramo repliku razumne dužine, ako postoji

SPAN_RE = re.compile(r"\[([^\[\]|]+)\|\s*([A-Z_]+)\s*\]")


def load_dialogs():
    dialogs = {}
    for path in (DATA / "osnovne_anotacije").glob("*.json"):
        with open(path, encoding="utf-8") as f:
            dialogs[int(path.stem)] = json.load(f)["annotations"]
    return dict(sorted(dialogs.items()))


def load_descriptions():
    """Opis svakog tipa iz Entity_types_description.md (engleski)."""
    desc = {}
    md = (DATA / "english_annotation" / "Entity_types_description.md").read_text(encoding="utf-8")
    for label, text in re.findall(r"\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*(.+?)\s*\|", md):
        desc[label] = text
    return desc


def plain(annotated, target=None):
    """Uklanja oznake; ciljni raspon (tekst, tip) označava sa »...«."""
    def repl(m):
        if target and (m.group(1).strip(), m.group(2)) == target:
            return f"»{m.group(1)}«"
        return m.group(1)
    return SPAN_RE.sub(repl, annotated)


dialogs = load_dialogs()
descriptions = load_descriptions()

# svi rasponi: (tip) -> lista (dialog_id, turn_idx, tekst raspona)
mentions = {}
for did, turns in dialogs.items():
    for i, turn in enumerate(turns):
        for span, label in SPAN_RE.findall(turn["annotated_text"]):
            mentions.setdefault(label, []).append((did, i, span.strip()))

lines = []
for label in sorted(mentions, key=lambda l: -len(mentions[l])):
    items = mentions[label]
    forms = {}
    for did, i, span in items:
        forms.setdefault(span.lower(), []).append((did, i, span))
    # najčešći oblici prvo
    ranked = sorted(forms.values(), key=len, reverse=True)

    lines.append("=" * 90)
    lines.append(f"{label}  ({len(items)} pojavljivanja, {len(forms)} različitih oblika)")
    lines.append(f"Opis: {descriptions.get(label, '-')}")
    lines.append("Najčešći oblici: " + ", ".join(f"{occ[0][2].lower()} ({len(occ)})" for occ in ranked[:5]))

    for occ in ranked[:N_PRIMERA]:
        def words(o):
            return len(dialogs[o[0]][o[1]]["text"].split())
        fitting = [o for o in occ if MIN_RECI <= words(o) <= MAX_RECI]
        did, idx, span = (fitting or occ)[0]
        turns = dialogs[did]
        lines.append(f"\nPrimer: »{span}«  (dijalog {did}, replika {idx + 1} od {len(turns)})")
        for j in range(max(0, idx - KONTEKST), min(len(turns), idx + KONTEKST + 1)):
            t = turns[j]
            target = (span, label) if j == idx else None
            marker = "→" if j == idx else " "
            lines.append(f"  {marker} {t['speaker']:<9}: {plain(t['annotated_text'], target)}")
    lines.append("")

report = "\n".join(lines)
print(report)
(OUT / "primeri_entiteta.txt").write_text(report, encoding="utf-8")
