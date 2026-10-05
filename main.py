"""
Faza 1 - Upoznavanje sa bazom "Halo beba".

Skripta učitava sva četiri izvora podataka, proverava njihovu međusobnu
konzistentnost i radi eksplorativnu analizu (EDA):
    1. konzistentnost izvora
    2. osnovni metapodaci (broj dijaloga, dužine, govornici, datumi)
    3. anotacije entiteta
    4. anotacije emocija
    5. raspon tema (na osnovu ključnih reči iz sažetaka)
    6. struktura razgovora (kako roditelji opisuju, kako sestre pitaju/savetuju)
    7. preliminarna ko-okurencija simptom-intervencija (uvod u Fazu 2)

Tekstualni izveštaj (izvestaj.txt) i tri grafika (tipovi entiteta, emocije,
teme) se čuvaju u folderu rezultati_faza1/.
"""
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # grafike samo čuvamo u fajl, ne otvaramo prozor
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "rezultati_faza1"
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 80)

# [tekst raspona|TIP] - dozvoljavamo razmak posle "|" (npr. "[measure| TEST]")
SPAN_RE = re.compile(r"\[([^\[\]|]+)\|\s*([A-Z_]+)\s*\]")
HAS_SPAN = r"\[[^\[\]|]+\|\s*[A-Z_]+\s*\]"  # isti obrazac bez grupa (za str.contains)
SPEAKER_MAP = {"CALLER": "POZIVALAC", "NURSE": "SESTRA"}

_report = open(OUT / "izvestaj.txt", "w", encoding="utf-8")


def log(*args):
    """Štampa na ekran i istovremeno upisuje u rezultati_faza1/izvestaj.txt."""
    text = " ".join(str(a) for a in args)
    print(text)
    _report.write(text + "\n")


def section(title):
    log("\n" + "=" * 90)
    log(title)
    log("=" * 90)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Učitavanje
# ---------------------------------------------------------------------------
# Sva četiri foldera sadrže ISTIH 150 dijaloga, samo različito anotiranih:
#   osnovne_anotacije/N.json                  -> entiteti (30 tipova)
#   anotacije_emocija/N_annotated.json        -> emocije (8 tipova)
#   summary_key_words_serbian/.../NNNs.json   -> transkript + sažetak + ključne reči
#   english_annotation/Corpus/N.json          -> engleski prevod sa entitetima
# Zato ih NE spajamo u jedan DataFrame (to bi utrostručilo svaku repliku),
# već svaki izvor učitavamo posebno i povezujemo ih preko dialog_id.

def dialog_id_from(path):
    return int(re.match(r"\d+", path.name).group())


def load_turns(folder, pattern, list_key):
    rows = []
    for path in sorted(folder.glob(pattern), key=dialog_id_from):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for i, turn in enumerate(data[list_key]):
            rows.append({
                "dialog_id": dialog_id_from(path),
                "turn_idx": i,
                "speaker": SPEAKER_MAP.get(turn["speaker"], turn["speaker"]),
                "text": turn["text"],
                "annotated_text": turn.get("annotated_text"),
            })
    df = pd.DataFrame(rows)
    n_turns = df.groupby("dialog_id")["turn_idx"].transform("size")
    # relativna pozicija replike u razgovoru: 0 = početak, 1 = kraj
    df["rel_pos"] = df["turn_idx"] / (n_turns - 1)
    df["n_words"] = df["text"].str.split().str.len()
    df["is_question"] = df["text"].str.contains(r"\?", regex=True)
    return df


def load_summaries(folder):
    rows = []
    for path in sorted(folder.glob("*.json"), key=dialog_id_from):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "dialog_id": dialog_id_from(path),
            "call_id": data["call_id"],
            "summary": " ".join(data["summary"]["text"]),
            "n_summary_sentences": len(data["summary"]["text"]),
            "key_words": [k.strip() for k in data["summary"]["key_words"]],
        })
    return pd.DataFrame(rows)


def extract_spans(turns):
    """Iz annotated_text vadi sve raspone [tekst|TIP] kao zasebne redove."""
    rows = []
    for t in turns.itertuples():
        for span, label in SPAN_RE.findall(t.annotated_text or ""):
            rows.append({
                "dialog_id": t.dialog_id,
                "turn_idx": t.turn_idx,
                "speaker": t.speaker,
                "rel_pos": t.rel_pos,
                "is_question": t.is_question,
                "span": span.strip(),
                "span_norm": span.strip().lower(),
                "label": label,
            })
    return pd.DataFrame(rows)


def strip_tags(annotated):
    return SPAN_RE.sub(lambda m: m.group(1), annotated)


turns = load_turns(DATA / "osnovne_anotacije", "*.json", "annotations")
emo_turns = load_turns(DATA / "anotacije_emocija", "*.json", "annotations")
sum_turns = load_turns(DATA / "summary_key_words_serbian" / "json_serbian", "*.json", "transcription")
en_turns = load_turns(DATA / "english_annotation" / "Corpus", "*.json", "annotations")
summaries = load_summaries(DATA / "summary_key_words_serbian" / "json_serbian")

entities = extract_spans(turns)
emotions = extract_spans(emo_turns)
en_entities = extract_spans(en_turns)

# ---------------------------------------------------------------------------
# 1. Konzistentnost izvora
# ---------------------------------------------------------------------------
section("1. KONZISTENTNOST IZVORA")

check = pd.DataFrame({
    "entiteti": turns.groupby("dialog_id").size(),
    "emocije": emo_turns.groupby("dialog_id").size(),
    "sazeci": sum_turns.groupby("dialog_id").size(),
    "engleski": en_turns.groupby("dialog_id").size(),
})
log("Broj dijaloga po izvoru:", check.notna().sum().to_dict())
log("Ukupno replika po izvoru:", check.sum().astype(int).to_dict())


def same_turn_share(other, did):
    """Udeo replika dijaloga čiji je tekst identičan tekstu iz osnovne_anotacije."""
    a = turns.loc[turns["dialog_id"] == did, "text"].tolist()
    b = other.loc[other["dialog_id"] == did, "text"].tolist()
    return round(sum(x == y for x, y in zip(a, b)) / max(len(a), len(b)), 3)


def text_similarity(other, did):
    """Prosečna sličnost (0-1) odgovarajućih replika - hvata sitne izmene teksta."""
    a = turns.loc[turns["dialog_id"] == did, "text"].tolist()
    b = other.loc[other["dialog_id"] == did, "text"].tolist()
    return round(sum(SequenceMatcher(None, x, y).ratio() for x, y in zip(a, b)) / max(len(a), len(b)), 3)


check["identicne_replike_sazeci"] = [same_turn_share(sum_turns, d) for d in check.index]
check["slicnost_sazeci"] = [text_similarity(sum_turns, d) for d in check.index]
check["slicnost_emocije"] = [text_similarity(emo_turns, d) for d in check.index]

diff_emo = check[check["emocije"] != check["entiteti"]]
diff_en = check[check["engleski"] != check["entiteti"]]
log(f"\nDijalozi gde se broj replika razlikuje (emocije vs entiteti): {len(diff_emo)}")
log(diff_emo[["entiteti", "emocije", "slicnost_emocije"]].to_string())
log(f"Ostali dijalozi u anotacijama emocija imaju prosečnu sličnost teksta "
    f"{check.drop(diff_emo.index)['slicnost_emocije'].mean():.3f}")
log(f"\nDijalozi gde engleski prevod ima drugačiji broj replika: {len(diff_en)} "
    "(prevodilac je spajao/delio replike - engleski korpus nije poravnat 1:1 po replikama)")
log(f"Transkripti u sažecima: {check['identicne_replike_sazeci'].mean():.1%} replika identično, "
    f"prosečna sličnost {check['slicnost_sazeci'].mean():.3f}, minimum {check['slicnost_sazeci'].min():.3f} "
    "(sitne redakcijske ispravke: red reči, interpunkcija)")

bad_strip = turns[turns["annotated_text"].map(strip_tags).str.strip() != turns["text"].str.strip()]
log(f"\nReplike gde annotated_text bez oznaka != text: {len(bad_strip)} "
    "(samo razlike u razmacima/interpunkciji, zanemarljivo)")
unknown = set(entities["label"]) - set(en_entities["label"])
log("Tipovi entiteta koji postoje u srpskom, a ne u engleskom korpusu:", unknown or "nema")

# ---------------------------------------------------------------------------
# 2. Osnovni metapodaci
# ---------------------------------------------------------------------------
section("2. OSNOVNI METAPODACI")

# call_id uglavnom sadrži vreme snimanja u milisekundama (001-record-1729582194641_ms),
# ali neki imaju drugačiji format (npr. 024_TamaraTijanaKašalj_ms - igrani/simulirani pozivi)
summaries["recorded_at"] = pd.to_datetime(
    pd.to_numeric(summaries["call_id"].str.extract(r"(\d{13})")[0]), unit="ms")
summaries["call_id_format"] = summaries["call_id"].str.replace(r"^\d+[-_]", "", regex=True).str.extract(
    r"^(record|renamed|[A-Za-zČĆŠĐŽčćšđž]+)")[0]

dialogs = turns.groupby("dialog_id").agg(
    n_turns=("turn_idx", "size"),
    n_words=("n_words", "sum"),
    first_speaker=("speaker", "first"),
    last_speaker=("speaker", "last"),
)
dialogs["n_entities"] = entities.groupby("dialog_id").size()
dialogs["n_emotions"] = emotions.groupby("dialog_id").size()
dialogs = dialogs.fillna({"n_emotions": 0}).astype({"n_emotions": int})
dialogs = dialogs.join(summaries.set_index("dialog_id"))

log(f"Broj dijaloga: {len(dialogs)}")
log(f"Broj replika:  {len(turns)}")
log(f"Broj reči:     {turns['n_words'].sum()}")
log(f"Period snimanja (za {summaries['recorded_at'].notna().sum()} dijaloga sa timestamp-om u call_id): "
    f"{summaries['recorded_at'].min():%d.%m.%Y} - {summaries['recorded_at'].max():%d.%m.%Y}")
log("Formati call_id:", summaries["call_id_format"].value_counts().to_dict())
log("\nBroj replika po dijalogu:")
log(dialogs["n_turns"].describe().round(1).to_string())
log("\nBroj reči po dijalogu:")
log(dialogs["n_words"].describe().round(1).to_string())

log("\nUdeo replika po govorniku (%):")
log((turns["speaker"].value_counts(normalize=True) * 100).round(1).to_string())
log("\nBroj reči po replici, po govorniku:")
log(turns.groupby("speaker")["n_words"].describe().round(1).to_string())
log("\nKo započinje razgovor:", dialogs["first_speaker"].value_counts().to_dict())
log("Ko završava razgovor:", dialogs["last_speaker"].value_counts().to_dict())

# naizmenično smenjivanje govornika (da li ima dve uzastopne replike istog govornika)
same_speaker = (turns["speaker"] == turns.groupby("dialog_id")["speaker"].shift()).sum()
log(f"Uzastopne replike istog govornika: {same_speaker} (razgovor je gotovo strogo naizmeničan)")

# ---------------------------------------------------------------------------
# 3. Anotacije entiteta
# ---------------------------------------------------------------------------
section("3. ANOTACIJE ENTITETA (osnovne_anotacije)")

log(f"Ukupno anotiranih raspona: {len(entities)}, tipova: {entities['label'].nunique()}")
log(f"Replike sa bar jednim entitetom: {turns['annotated_text'].str.contains(HAS_SPAN).mean():.1%}")

label_stats = pd.DataFrame({
    "broj": entities["label"].value_counts(),
    "u_dijaloga": entities.groupby("label")["dialog_id"].nunique(),
    "jedinstvenih_oblika": entities.groupby("label")["span_norm"].nunique(),
    "udeo_sestra_%": (entities.groupby("label")["speaker"]
                      .apply(lambda s: (s == "SESTRA").mean() * 100).round(1)),
}).sort_values("broj", ascending=False)
label_stats["engleski_broj"] = en_entities["label"].value_counts()
log(label_stats.to_string())

key_labels = ["SYMPTOM", "INTERVENTION", "TREATMENT", "EVENT", "OUTCOME", "BEHAVIOR"]
for label in key_labels:
    top = entities.loc[entities["label"] == label, "span_norm"].value_counts().head(15)
    log(f"\nNajčešći oblici za {label}:")
    log(", ".join(f"{s} ({n})" for s, n in top.items()))

# Isti pojam se javlja u mnogo oblika (padeži, sinonimi, fraze) -> potrebna normalizacija
temp_forms = entities.loc[entities["span_norm"].str.contains("temperatur"), "span_norm"].value_counts()
log(f"\nPrimer problema normalizacije - raspona koji sadrže 'temperatur': {temp_forms.sum()} "
    f"pojavljivanja u {len(temp_forms)} različitih oblika:")
log(", ".join(f"{s} ({n})" for s, n in temp_forms.head(15).items()))

span_len = entities["span"].str.split().str.len()
log("\nDužina raspona (broj reči) po tipu - medijana / max:")
log(span_len.groupby(entities["label"]).agg(["median", "max"]).loc[key_labels].to_string())

label_stats["broj"].sort_values().plot.barh(figsize=(7, 8), color="#4C72B0")
plt.title("Broj anotacija po tipu entiteta")
plt.ylabel("")
save_fig("02_tipovi_entiteta.png")

# ---------------------------------------------------------------------------
# 4. Anotacije emocija
# ---------------------------------------------------------------------------
section("4. ANOTACIJE EMOCIJA / SENTIMENTA (anotacije_emocija)")

log(f"Ukupno anotiranih raspona: {len(emotions)}, "
    f"u {emotions['dialog_id'].nunique()} od {emo_turns['dialog_id'].nunique()} dijaloga")
log(f"Replike sa emocijom: {emo_turns['annotated_text'].str.contains(HAS_SPAN).mean():.1%}")
log("\nEmocije po govorniku:")
emo_by_speaker = pd.crosstab(emotions["label"], emotions["speaker"], margins=True, margins_name="ukupno")
log(emo_by_speaker.sort_values("ukupno", ascending=False).to_string())

log("\nPrimeri raspona po emociji:")
for label, group in emotions.groupby("label"):
    examples = group["span"].drop_duplicates().head(3).tolist()
    log(f"  {label}: " + " | ".join(examples))

# UNCERTAIN (emocije) i UNCERTAINTY (entiteti) su dva nezavisna označavanja iste pojave
unc_emo = set(emotions.loc[emotions["label"] == "UNCERTAIN", ["dialog_id", "turn_idx"]].itertuples(index=False))
unc_ent = set(entities.loc[entities["label"] == "UNCERTAINTY", ["dialog_id", "turn_idx"]].itertuples(index=False))
log(f"\nReplike sa UNCERTAIN (emocije): {len(unc_emo)}, sa UNCERTAINTY (entiteti): {len(unc_ent)}, "
    f"presek: {len(unc_emo & unc_ent)}")

emotions["label"].value_counts().sort_values().plot.barh(figsize=(7, 4), color="#DD8452")
plt.title("Broj anotacija po emociji")
plt.ylabel("")
save_fig("03_emocije.png")

# ---------------------------------------------------------------------------
# 5. Raspon tema
# ---------------------------------------------------------------------------
section("5. RASPON TEMA (ključne reči iz sažetaka = razlog poziva)")

# Ručno definisane kategorije (regex nad ključnim rečima sažetka, koje u jednoj
# frazi opisuju razlog poziva). Dijalog može pripadati većem broju tema.
TOPICS = {
    "temperatura": r"temperatur|febril",
    "koža (osip, crvenilo, promene)": r"osip|osp[ea]|crven|plikov|kož|perut|naslag|tačk|ekcem|vašk|ljusp|češe|ojed",
    "stolica / probava / stomak": r"stolic|proliv|dijarej|kaki|grčev|stomak",
    "povraćanje / bljuckanje": r"povra|bljuc|izbacuje puno mleka",
    "disajni putevi / nos / kašalj": r"kašl|kašalj|\bnos|diš|kijav|krajnik",
    "ishrana / dojenje": r"\bjede|sis[au]|mlek|kašic|hran|doj|vodu ili čaj|cokće|probirljiv",
    "spavanje / plač / ponašanje": r"spav|budi|plač|plak|plače|vrišti|razdražlj|uspavan|trza|trgne|štuca|zacenj|prisutn|besa|drsk|grize|pažnj|pokriven|dozivao",
    "povrede / nezgode": r"\bpal[aoie]\b|povred|progut|kliker|igračk|\brane\b|krvari|bol u obe noge|koristi ruku",
    "oči / uši": r"krmelj|suz|očn|\boka\b|očima|\buh[ao]\b|\buši|ušima|čuje",
    "pupak": r"pup[akč]",
    "razvoj i navike": r"govor|priča|pelena\b|cucl|sedi|stoji|okreće glavu|\bgleda|vrtić|sport",
    "vakcinacija": r"vakcin",
    "mokrenje": r"urin|mokr",
    "zdravlje majke / dojilje": r"dojci|porodilj|svojom ishranom|dovoljno mleka|prestane sa dojenjem|porođaja",
    "putovanje / šetnja / priprema": r"putovanj|šetnju|operacij",
}

dialogs["topic_text"] = dialogs["key_words"].apply(lambda kw: " ".join(kw).lower())
dialogs["topics"] = dialogs["topic_text"].apply(
    lambda t: [name for name, pattern in TOPICS.items() if re.search(pattern, t)] or ["ostalo"])

topic_counts = dialogs["topics"].explode().value_counts()
log("Broj dijaloga po temi (dijalog može imati više tema):")
log(topic_counts.to_string())
log(f"\nProsečan broj tema po dijalogu: {dialogs['topics'].str.len().mean():.2f}")
log("\nDijalozi svrstani u 'ostalo' (proveriti ručno):")
for did, row in dialogs[dialogs["topics"].apply(lambda t: t == ['ostalo'])].iterrows():
    log(f"  {did}: {'; '.join(row['key_words'])}")

topic_counts.sort_values().plot.barh(figsize=(8, 5), color="#55A868")
plt.title("Broj dijaloga po temi")
plt.ylabel("")
save_fig("04_teme.png")

# ---------------------------------------------------------------------------
# 6. Struktura razgovora
# ---------------------------------------------------------------------------
section("6. STRUKTURA RAZGOVORA")

log("Udeo replika koje sadrže pitanje (%):")
log((turns.groupby("speaker")["is_question"].mean() * 100).round(1).to_string())

# kako roditelj otvara problem: prva replika pozivaoca koja sadrži simptom/ponašanje
first_complaint = (entities[(entities["speaker"] == "POZIVALAC")
                            & entities["label"].isin(["SYMPTOM", "BEHAVIOR", "STOOL", "EVENT"])]
                   .groupby("dialog_id")["turn_idx"].min())
log(f"\nPrva replika u kojoj pozivalac pominje problem (SYMPTOM/BEHAVIOR/STOOL/EVENT) - "
    f"medijana indeksa: {first_complaint.median():.0f}; u prve 3 replike: {(first_complaint <= 2).mean():.0%}")
log("Primeri prvih opisa problema:")
for did in [1, 17, 45, 100, 131]:
    t = turns[(turns["dialog_id"] == did) & (turns["speaker"] == "POZIVALAC")]
    t = t[t["annotated_text"].str.contains(r"\|(?:SYMPTOM|BEHAVIOR|STOOL|EVENT)\]")].head(1)
    for row in t.itertuples():
        log(f"  [{did}] {row.annotated_text}")

# kako sestra postavlja follow-up pitanja: prva reč pitanja
nurse_q = turns[(turns["speaker"] == "SESTRA") & turns["is_question"]]
q_word = (nurse_q["text"].str.extract(r"([^?.!,]*\?)")[0]
          .str.lower().str.extract(r"(da li|kako|koliko|kada|kad|gde|šta|koji|koja|koje|zašto|od kada|je l)")[0])
log(f"\nSestrinska pitanja ({len(nurse_q)}) - početni upitni izraz:")
log((q_word.fillna("(ostalo)").rename("upitni izraz").value_counts(normalize=True) * 100).round(1).head(12).to_string())

log("\nO čemu sestra pita - tipovi entiteta u sestrinim pitanjima (%):")
nurse_q_ent = entities[(entities["speaker"] == "SESTRA") & entities["is_question"]]
log((nurse_q_ent["label"].value_counts(normalize=True) * 100).round(1).head(12).to_string())

# gde se u razgovoru javljaju tipovi entiteta: početak / sredina / kraj
entities["deo_razgovora"] = pd.cut(entities["rel_pos"], [-0.01, 1 / 3, 2 / 3, 1.0],
                                   labels=["početak", "sredina", "kraj"])
pos = pd.crosstab([entities["label"], entities["speaker"]], entities["deo_razgovora"], normalize="index")
log("\nPozicija entiteta u razgovoru (udeo po trećinama):")
log((pos.loc[key_labels] * 100).round(0).to_string())

# Intervencije/terapije: pokušane (pozivalac) vs pitanje sestre vs savet sestre
iv = entities[entities["label"].isin(["INTERVENTION", "TREATMENT"])].copy()
iv["uloga"] = "pozivalac (pokušano/opisano)"
iv.loc[(iv["speaker"] == "SESTRA") & iv["is_question"], "uloga"] = "sestra - pitanje (da li ste probali...)"
iv.loc[(iv["speaker"] == "SESTRA") & ~iv["is_question"], "uloga"] = "sestra - izjava (savet/preporuka)"
log("\nINTERVENTION + TREATMENT prema ulozi u razgovoru:")
log(pd.crosstab(iv["uloga"], iv["label"], margins=True, margins_name="ukupno").to_string())
log("\nPrimeri sestrinih saveta (izjave sa INTERVENTION/TREATMENT u poslednjoj trećini):")
advice = iv[(iv["uloga"].str.contains("savet")) & (iv["rel_pos"] > 2 / 3)]
for (did, tidx) in advice[["dialog_id", "turn_idx"]].drop_duplicates().head(6).itertuples(index=False):
    log(f"  [{did}] " + turns.loc[(turns['dialog_id'] == did) & (turns['turn_idx'] == tidx), 'annotated_text'].iloc[0])

# ---------------------------------------------------------------------------
# 7. Preliminarna ko-okurencija simptom - intervencija (uvod u Fazu 2)
# ---------------------------------------------------------------------------
section("7. PRELIMINARNA KO-OKURENCIJA SIMPTOM - INTERVENCIJA/TERAPIJA")

# Na nivou replike ko-okurencija je retka jer pozivalac opiše simptom u jednoj,
# a intervenciju u nekoj kasnijoj replici. Zato poredimo oba nivoa.
sym = entities[entities["label"] == "SYMPTOM"]
act = entities[entities["label"].isin(["INTERVENTION", "TREATMENT"])]

turn_pairs = sym.merge(act, on=["dialog_id", "turn_idx"], suffixes=("_s", "_i"))
dialog_pairs = (sym[["dialog_id", "span_norm"]].drop_duplicates()
                .merge(act[["dialog_id", "span_norm", "label"]].drop_duplicates(),
                       on="dialog_id", suffixes=("_s", "_i")))
log(f"Parova simptom-intervencija u istoj replici: {len(turn_pairs)} "
    f"(u {turn_pairs['dialog_id'].nunique()} dijaloga)")
log(f"Parova simptom-intervencija u istom dijalogu: {len(dialog_pairs)} "
    f"(u {dialog_pairs['dialog_id'].nunique()} dijaloga)")

log("\nNajčešći parovi na nivou dijaloga (broj dijaloga):")
top_pairs = (dialog_pairs.groupby(["span_norm_s", "span_norm_i"])["dialog_id"].nunique()
             .sort_values(ascending=False).head(15))
log(top_pairs.to_string())
log("\nNAPOMENA: parovi su i dalje fragmentisani jer isti pojam ima mnogo oblika "
    "('temperatura', 'temperaturu', 'povišenu temperaturu'...). Lematizacija i spajanje "
    "sinonima (Faza 2/3) su neophodni pre izgradnje grafa.")

log(f"\nRezultati sačuvani u: {OUT}")
_report.close()
