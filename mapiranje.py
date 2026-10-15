"""
Faza 2 - Mapiranje 30 tipova entiteta na klase ontologije (pravila).

Ulaz je isti za ručne anotacije i za izlaz BERTić-a: replike kao tokeni + BIO oznake.
    gold: data_ner/train.jsonl, dev.jsonl, test.jsonl   (polje ner_tags)
    NER:  data_ner/predikcije_test.jsonl                (polje pred_tags, iz Colab-a)

Klase:
    čvorovi znanja:  SIMPTOM, INTERVENCIJA, TERAPIJA, UZROK, ISHOD
    ATRIBUT_DIJALOGA     uzrast, vrtić, vakcinacija, osobine deteta -> svojstva čvora Dijalog
    ATRIBUT_POMINJANJA   vreme, trajanje, učestalost, deo tela... -> kači se na najbliži čvor znanja
    ODBACENO             ne ulazi u graf

Svako pominjanje dobija i naziv pravila koje ga je razvrstalo, da bi se odluke mogle proveriti.
Klasa PROBLEM se ne anotira - problemi se izvode iz zajednica u fazi 4.

Izlaz (rezultati_mapiranje/, ne ide na GitHub jer sadrži tekst dijaloga):
    pominjanja_gold.csv   sva pominjanja iz ručnih anotacija sa klasom i pravilom
    za_anotaciju.csv      pominjanja iz test dijaloga koja zavise od konteksta -> ručna provera
    zbir_mapiranja.txt    tabela tip -> klasa i evaluacija na nivou klasa (samo brojke)
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_ner"
OUT = BASE / "rezultati_mapiranje"
OUT.mkdir(exist_ok=True)
PRED_FILE = DATA / "predikcije_test.jsonl"

ZNANJE = ["SIMPTOM", "INTERVENCIJA", "TERAPIJA", "UZROK", "ISHOD"]

# Tipovi čija klasa ne zavisi od konteksta
FIKSNO = {
    "SYMPTOM": "SIMPTOM",
    "INTERVENTION": "INTERVENCIJA",
    "TEST": "INTERVENCIJA",          # podtip: provera
    "TREATMENT": "TERAPIJA",
    "EVENT": "UZROK",
    "MOTHER_NUTRITION": "UZROK",
    "AGE": "ATRIBUT_DIJALOGA",
    "KINDERGARTEN": "ATRIBUT_DIJALOGA",
    "MEDICAL_DEVICE": "ATRIBUT_POMINJANJA",  # sredstvo intervencije
    "TIME": "ATRIBUT_POMINJANJA",
    "DURATION": "ATRIBUT_POMINJANJA",
    "FREQUENCY": "ATRIBUT_POMINJANJA",
    "QUANTITY": "ATRIBUT_POMINJANJA",
    "DEGREE": "ATRIBUT_POMINJANJA",
    "CHARACTERISTIC": "ATRIBUT_POMINJANJA",
    "BODY_PART": "ATRIBUT_POMINJANJA",
    "UNCERTAINTY": "ATRIBUT_POMINJANJA",
    "ABNORMALITY": "ODBACENO",
    "CAREGIVER_SUPPORT": "ODBACENO",
    "CLOTHES": "ODBACENO",
}
# Tipovi koje razvrstavaju pravila ispod (zavise od konteksta)
KONTEKSTNI = ["BEHAVIOR", "FEEDING", "STOOL", "URINATION", "HYDRATION", "CHILD_TRAIT",
              "ENVIRONMENT", "FOOD", "VACCINATION_STATUS", "OUTCOME"]

# Reči (početci reči) koje ukazuju na odstupanje od normalnog
ODSTUPANJE = ("problem", "odbij", "slab", "manje", "previše", "preterano", "teško", "tvrd",
              "retk", "rijetk", "redj", "ređ", "zelen", "sluz", "krv", "nemir", "plač", "plak",
              "razdraž", "trza", "trgn", "grč", "zgrč", "ukoč", "kruć", "oslab", "bled", "bleđ",
              "tamn", "smrd", "čudn", "neobič", "promen", "povrać", "bljuc", "dijarej", "proliv",
              "zatvor", "zaosta", "nervoz", "uznemir", "boli", "bolov", "bolan", "bolna", "bolno", "muči", "muka", "neredov", "stalno")
NEGACIJA = {"ne", "nije", "nisu", "neće", "nece", "nema", "nemaju", "ni", "nikako", "nimalo"}
# Savet sestre: eksplicitne reči + zapovedni način (-ajte, -ite, -ujte), osim fraza iz razgovora
SAVET = ("savet", "preporuč", "treba", "trebalo", "morate", "možete", "obavezno", "nemojte",
         "izbegava")
NIJE_SAVET = {"recite", "kažite", "izvolite", "sačekajte", "izvinite", "oprostite", "slušajte",
              "znate", "imate", "jeste", "ste", "hvala", "molite", "čujete"}
UZROCNO = ("zbog", "posle", "nakon", "otkad", "otkako", "kontakt", "izazva", "uzrok", "reakcij",
           "reagu", "alergi", "smeta", "prija", "poveza", "povezu", "od kad")
PROZOR = 3  # koliko reči levo i desno od entiteta gledamo pri traženju reči-okidača


# ---------------------------------------------------------------------------
# Učitavanje
# ---------------------------------------------------------------------------
def load(paths, tag_field):
    dialogs = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dialogs[r["dialog_id"]].append(
                    {"speaker": r["speaker"], "tokens": r["tokens"], "tags": r[tag_field]})
    return dict(sorted(dialogs.items()))

# cita replike
# svaka replika postaje rečnik {speaker, tokens, tags} i dodaje se u listu svog dijaloga.
# dijalozi se sortiraju
# Rezultat izgleda ovako: {1: [replika0, replika1, …], 2: […], …}.

def spans_of(tags):
    """Entiteti iz BIO niza (strogi IOB2: I- bez odgovarajućeg B- se odbacuje)."""
    spans, start, label = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if start is not None and tag != "I-" + label:
            spans.append((start, i, label))
            start = None
        if tag.startswith("B-"):
            start, label = i, tag[2:]
    return spans

# Primer:(3, 7, "SYMPTOM") → tokeni 3–6:  "neko crvenilo po vratu"

# ---------------------------------------------------------------------------
# Pomoćne funkcije za kontekst
# ---------------------------------------------------------------------------
def sentence_bounds(tokens, s, e):
    """Granice rečenice u kojoj je entitet [s, e)."""
    left = s
    while left > 0 and tokens[left - 1] not in ".?!":
        left -= 1
    right = e
    while right < len(tokens) and tokens[right] not in ".?!":
        right += 1
    return left, min(right + 1, len(tokens))


def is_question(tokens, s, e):
    left, right = sentence_bounds(tokens, s, e)
    return tokens[right - 1] == "?"

# Vraća reči entiteta ± 3 reči, malim slovima, ali ne prelazi granicu rečenice: max(left, s-n) i min(right, e+n).

def window(tokens, s, e, n=PROZOR):
    """Reči entiteta + n reči levo i desno, ali unutar iste rečenice."""
    left, right = sentence_bounds(tokens, s, e)
    return [t.lower() for t in tokens[max(left, s - n):min(right, e + n)]]

# od kad
def has_prefix(words, prefixes):
    text = " ".join(words)
    return any(w.startswith(prefixes) for w in words) or any(" " in p and p in text for p in prefixes)

# proverava da li je savet sestre
def is_advice(tokens, s, e, speaker):
    if speaker != "SESTRA" or is_question(tokens, s, e):
        return False
    left, right = sentence_bounds(tokens, s, e)
    words = [t.lower() for t in tokens[left:right]]
    imperative = any(w.endswith(("ajte", "ite", "ujte", "ejte")) and len(w) > 4
                     and w not in NIJE_SAVET for w in words)
    return imperative or has_prefix(words, SAVET)

# da li opisuje neko odstupanje
def is_abnormal(tokens, s, e, other_spans):
    words = window(tokens, s, e)
    if has_prefix(words, ODSTUPANJE) or NEGACIJA & set(words):
        return True
    # u istoj rečenici je i simptom ili oznaka problema (npr. "problem sa stolicom")
    left, right = sentence_bounds(tokens, s, e)
    return any(lab in ("SYMPTOM", "ABNORMALITY") and left <= a < right for a, b, lab in other_spans)

# uzor je ako se postoji neka uzorcna rec uz entitet ili ako postoji uzorcna rec i simptom ili odustanje
def is_causal(tokens, s, e, other_spans):
    left, right = sentence_bounds(tokens, s, e)
    words = [t.lower() for t in tokens[left:right]]
    return (has_prefix(words, UZROCNO) and any(
        lab in ("SYMPTOM", "STOOL", "BEHAVIOR", "ABNORMALITY") for a, b, lab in other_spans
        if left <= a < right)) or has_prefix(window(tokens, s, e), UZROCNO)


# ---------------------------------------------------------------------------
# Pravila
# ---------------------------------------------------------------------------
def classify(label, tokens, s, e, speaker, spans, prev_labels):
    """Vraća (klasa, naziv pravila)."""
    if label in FIKSNO:
        return FIKSNO[label], "fiksno"
    others = [x for x in spans if x != (s, e, label)]

    if label in ("BEHAVIOR", "FEEDING", "HYDRATION", "URINATION", "STOOL"):
        if label in ("FEEDING", "HYDRATION") and is_advice(tokens, s, e, speaker):
            return "INTERVENCIJA", "savet sestre"
        if is_abnormal(tokens, s, e, others):
            return "SIMPTOM", "odstupanje"
        return "ODBACENO", "opis normalnog stanja"

    if label == "CHILD_TRAIT":
        if is_abnormal(tokens, s, e, others):
            return "SIMPTOM", "odstupanje"
        return "ATRIBUT_DIJALOGA", "osobina deteta"

    if label == "ENVIRONMENT":
        if is_advice(tokens, s, e, speaker):
            return "INTERVENCIJA", "savet sestre"
        if is_causal(tokens, s, e, others):
            return "UZROK", "uzročna veza"
        return "ODBACENO", "okruženje bez uzročne veze"

    if label == "FOOD":
        if is_causal(tokens, s, e, others):
            return "UZROK", "uzročna veza"
        return "ATRIBUT_POMINJANJA", "hrana kao objekat ishrane"

    if label == "VACCINATION_STATUS":
        if is_causal(tokens, s, e, others):
            return "UZROK", "uzročna veza"
        return "ATRIBUT_DIJALOGA", "status vakcinacije"

    if label == "OUTCOME":
        # ishod ima smisla samo kao rezultat nečega što je urađeno (u ovoj ili prethodne 2 replike)
        if {"INTERVENTION", "TREATMENT", "TEST"} & ({lab for _, _, lab in spans} | prev_labels):
            return "ISHOD", "posle intervencije/terapije"
        return "ODBACENO", "opšte stanje, nije rezultat radnje"

    return "ODBACENO", "nepoznat tip"


def role(klasa, label, tokens, s, e, speaker):
    """Uloga intervencije/terapije u razgovoru -> kasnije tip grane (POKUSANO_KOD / PREPORUCENO_KOD)."""
    if klasa not in ("INTERVENCIJA", "TERAPIJA"):
        return ""
    if speaker == "POZIVALAC":
        return "pokušano"
    if is_question(tokens, s, e):
        return "pitanje"
    return "preporuka" if is_advice(tokens, s, e, speaker) else "pominjanje"


def prefer(attr, nodes):
    """Medicinsko sredstvo se kači na intervenciju/terapiju ako postoji, ostali atributi na bilo koji čvor."""
    if attr["tip"] == "MEDICAL_DEVICE":
        return [n for n in nodes if n["klasa"] in ("INTERVENCIJA", "TERAPIJA")] or nodes
    return nodes


def map_dialog(dialog_id, turns):
    mentions, prev_nodes = [], []
    for i, turn in enumerate(turns):
        tokens, speaker = turn["tokens"], turn["speaker"]
        spans = spans_of(turn["tags"])
        prev_labels = {lab for t in turns[max(0, i - 2):i] for _, _, lab in spans_of(t["tags"])}
        turn_mentions = []
        for s, e, label in spans:
            klasa, pravilo = classify(label, tokens, s, e, speaker, spans, prev_labels)
            turn_mentions.append({
                "dialog_id": dialog_id, "replika": i, "govornik": speaker, "tip": label,
                "tekst": " ".join(tokens[s:e]), "start": s, "end": e,
                "klasa": klasa, "pravilo": pravilo,
                "pitanje": is_question(tokens, s, e),
                "uloga": role(klasa, label, tokens, s, e, speaker),
                "pripada": "",
            })
        # atribut pominjanja se kači na najbliži čvor znanja u istoj replici;
        # ako ga tu nema, na poslednji čvor iz prethodne replike (pitanje sestre -> odgovor roditelja)
        nodes = [m for m in turn_mentions if m["klasa"] in ZNANJE]
        for m in turn_mentions:
            if m["klasa"] != "ATRIBUT_POMINJANJA":
                continue
            if nodes:
                near = min(prefer(m, nodes), key=lambda n: min(abs(n["start"] - m["end"]), abs(m["start"] - n["end"])))
                m["pripada"] = f"{near['klasa']}: {near['tekst']}"
            elif prev_nodes:
                near = max(prefer(m, prev_nodes), key=lambda n: n["start"])
                m["pripada"] = f"{near['klasa']}: {near['tekst']} (prethodna replika)"
        mentions.extend(turn_mentions)
        prev_nodes = nodes
    return mentions


def map_all(dialogs):
    return [m for did, turns in dialogs.items() for m in map_dialog(did, turns)]


# ---------------------------------------------------------------------------
# Evaluacija na nivou klasa: gold-mapirano vs NER-mapirano
# ---------------------------------------------------------------------------
def class_eval(gold_m, pred_m):
    """Po klasi: stroga mera (iste granice) i blaga (preklapanje u istoj replici)."""
    def key(m):
        return (m["dialog_id"], m["replika"], m["klasa"])
    rows = []
    for klasa in ZNANJE + ["UKUPNO"]:
        G = [m for m in gold_m if m["klasa"] == klasa or (klasa == "UKUPNO" and m["klasa"] in ZNANJE)]
        P = [m for m in pred_m if m["klasa"] == klasa or (klasa == "UKUPNO" and m["klasa"] in ZNANJE)]
        g_idx, p_idx = defaultdict(list), defaultdict(list)
        for m in G:
            g_idx[key(m)].append(m)
        for m in P:
            p_idx[key(m)].append(m)
        res = {"klasa": klasa, "gold": len(G), "ner": len(P)}
        for mode in ("stroga", "blaga"):
            def match(a, b):
                if mode == "stroga":
                    return (a["start"], a["end"]) == (b["start"], b["end"])
                return a["start"] < b["end"] and b["start"] < a["end"]
            hit_p = sum(any(match(p, g) for g in g_idx[key(p)]) for p in P)
            hit_g = sum(any(match(g, p) for p in p_idx[key(g)]) for g in G)
            prec = hit_p / len(P) if P else 0
            rec = hit_g / len(G) if G else 0
            res[f"F1_{mode}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0
            res[f"P_{mode}"], res[f"R_{mode}"] = prec, rec
        rows.append(res)
    return rows


# ---------------------------------------------------------------------------
# Glavni deo
# ---------------------------------------------------------------------------
FIELDS = ["dialog_id", "replika", "govornik", "tip", "tekst", "klasa", "pravilo", "pitanje", "uloga", "pripada"]
split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
gold = load([DATA / f"{n}.jsonl" for n in ("train", "dev", "test")], "ner_tags")
gold_m = map_all(gold)

with open(OUT / "pominjanja_gold.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(gold_m)

lines = []
# 1. Tabela tip -> klasa
cross = defaultdict(Counter)
for m in gold_m:
    cross[m["tip"]][m["klasa"]] += 1
classes = ZNANJE + ["ATRIBUT_DIJALOGA", "ATRIBUT_POMINJANJA", "ODBACENO"]
short = ["SIMPT", "INTERV", "TERAP", "UZROK", "ISHOD", "A_DIJ", "A_POM", "ODB"]
lines.append(f"Mapiranje ručnih anotacija ({len(gold)} dijaloga, {len(gold_m)} pominjanja)\n")
lines.append(f"{'tip':<20}" + "".join(f"{c:>8}" for c in short))
for tip in sorted(cross, key=lambda t: -sum(cross[t].values())):
    lines.append(f"{tip:<20}" + "".join(f"{cross[tip][c] or '':>8}" for c in classes))
total = Counter(m["klasa"] for m in gold_m)
lines.append(f"{'UKUPNO':<20}" + "".join(f"{total[c]:>8}" for c in classes))

# 2. Uloge intervencija i terapija
roles = Counter((m["klasa"], m["uloga"]) for m in gold_m if m["uloga"])
lines.append("\nUloga intervencija/terapija (-> tip grane):")
for (k, u), n in sorted(roles.items()):
    lines.append(f"  {k:<14}{u:<12}{n:>6}")
lines.append(f"\nPominjanja čvorova znanja u pitanjima sestre: "
             f"{sum(m['pitanje'] and m['govornik'] == 'SESTRA' for m in gold_m if m['klasa'] in ZNANJE)}")

# 3. Evaluacija na nivou klasa (ako postoje predikcije iz Colab-a)
if PRED_FILE.exists():
    pred = load([PRED_FILE], "pred_tags")
    pred_m = map_all(pred)
    gold_test_m = [m for m in gold_m if m["dialog_id"] in pred]
    lines.append(f"\nEvaluacija na nivou klasa: ručne anotacije vs BERTić, oba kroz ista pravila "
                 f"({len(pred)} test dijaloga)")
    lines.append(f"{'klasa':<14}{'gold':>6}{'ner':>6}{'P':>8}{'R':>8}{'F1':>8}{'P_bl':>8}{'R_bl':>8}{'F1_bl':>8}")
    for r in class_eval(gold_test_m, pred_m):
        lines.append(f"{r['klasa']:<14}{r['gold']:>6}{r['ner']:>6}"
                     f"{r['P_stroga']:>8.3f}{r['R_stroga']:>8.3f}{r['F1_stroga']:>8.3f}"
                     f"{r['P_blaga']:>8.3f}{r['R_blaga']:>8.3f}{r['F1_blaga']:>8.3f}")
    lines.append("(stroga = iste granice; blaga = preklapanje u istoj replici; obe traže istu klasu)")
else:
    lines.append(f"\n(Nema {PRED_FILE.name} -> preskačem evaluaciju BERTić-a na nivou klasa)")

report = "\n".join(lines)
print(report)
(OUT / "zbir_mapiranja.txt").write_text(report, encoding="utf-8")

# 4. Fajl za ručnu proveru: test dijalozi, samo tipovi čija klasa zavisi od konteksta
ann_path = OUT / "za_anotaciju.csv"
if ann_path.exists():
    print(f"\n{ann_path.name} već postoji - ne prepisujem ga (možda je već popunjen).")
else:
    test_ids = set(split["test"])
    rows = []
    for m in gold_m:
        if m["dialog_id"] in test_ids and m["tip"] in KONTEKSTNI:
            turns = gold[m["dialog_id"]]
            toks = turns[m["replika"]]["tokens"]
            prev = turns[m["replika"] - 1] if m["replika"] > 0 else None
            rows.append({**m,
                         "prethodna_replika": f"{prev['speaker']}: {' '.join(prev['tokens'])}" if prev else "",
                         "replika_tekst": " ".join(toks[:m["start"]] + ["»"] + toks[m["start"]:m["end"]]
                                                   + ["«"] + toks[m["end"]:]),
                         "klasa_rucno": ""})
    with open(ann_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["dialog_id", "replika", "govornik", "tip", "tekst",
                                          "prethodna_replika", "replika_tekst", "klasa", "pravilo",
                                          "klasa_rucno"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nZa ručnu proveru: {len(rows)} pominjanja iz {len(test_ids)} test dijaloga -> {ann_path}")

# 5. Poređenje pravila sa ručnom proverom (kada je kolona klasa_rucno popunjena)
#    U kolonu se upisuje "ok" ako je pravilo pogodilo, ili tačna klasa ako nije.
with open(ann_path, encoding="utf-8-sig") as f:
    checked = [r for r in csv.DictReader(f) if r["klasa_rucno"].strip()]
if checked:
    for r in checked:
        r["tacno"] = r["klasa_rucno"].strip().upper() in ("OK", r["klasa"])
    lines = [f"\nPravila naspram ručne provere ({len(checked)} pominjanja): "
             f"tačnost {sum(r['tacno'] for r in checked) / len(checked):.3f}",
             f"{'tip':<20}{'n':>5}{'tačno':>8}"]
    for tip in KONTEKSTNI:
        rs = [r for r in checked if r["tip"] == tip]
        if rs:
            lines.append(f"{tip:<20}{len(rs):>5}{sum(r['tacno'] for r in rs) / len(rs):>8.3f}")
    errors = Counter((r["tip"], r["klasa"], r["klasa_rucno"].strip().upper()) for r in checked if not r["tacno"])
    lines.append("Najčešće greške pravila (tip: pravilo -> ručno):")
    lines += [f"  {n:>4}  {t}: {k} -> {h}" for (t, k, h), n in errors.most_common(10)]
    print("\n".join(lines))
    with open(OUT / "zbir_mapiranja.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
