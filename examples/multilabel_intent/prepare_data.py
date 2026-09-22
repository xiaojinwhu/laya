"""Build multi-label intent datasets in the JSONL format `laya.multilabel` trains on.

    python prepare_data.py mixsnips --out data/mixsnips
    python prepare_data.py mixatis  --out data/mixatis
    python prepare_data.py massive  --out data/massive_zh --lang zh-CN

mixsnips / mixatis  the multi-intent SLU benchmarks of Qin et al. (AGIF, 2020), 1-3 intents each
massive             multi-intent utterances composed from MASSIVE (51 languages, 60 intents) the
                    way MixSNIPS was composed from SNIPS: 1-3 single-intent utterances with
                    different intents, joined by a connective

Each writes train.jsonl / dev.jsonl / test.jsonl ({"text": ..., "labels": [...]}) and labels.json.
"""
import argparse
import gzip
import json
import os
import random
import urllib.request

AGIF = "https://raw.githubusercontent.com/LooperXX/AGIF/master/data/%s/%s.txt"

DESCRIPTIONS = {
    "mixsnips": {
        "AddToPlaylist": "add a song, album or artist to a playlist",
        "BookRestaurant": "reserve a table at a restaurant, bar or cafe",
        "GetWeather": "ask about the weather or a forecast",
        "PlayMusic": "play a song, album, artist or playlist",
        "RateBook": "give a rating to a book",
        "SearchCreativeWork": "find a book, movie, song, show or game by its title",
        "SearchScreeningEvent": "find movie showtimes or what is playing at a cinema",
    },
    "mixatis": {
        "atis_abbreviation": "meaning of an airline, fare or airport code",
        "atis_aircraft": "type of aircraft used on a flight",
        "atis_airfare": "ticket prices and fares",
        "atis_airline": "which airlines fly a route",
        "atis_airport": "airports in or near a city",
        "atis_capacity": "seating capacity of an aircraft",
        "atis_cheapest": "the cheapest fare",
        "atis_city": "which city an airport or airline serves",
        "atis_day_name": "which days of the week a flight operates",
        "atis_distance": "distance between an airport and a city",
        "atis_flight": "flights between places, schedules and availability",
        "atis_flight_no": "flight numbers",
        "atis_flight_time": "departure or arrival times of flights",
        "atis_ground_fare": "cost of ground transportation",
        "atis_ground_service": "ground transportation at a city or airport",
        "atis_meal": "meals served on a flight",
        "atis_quantity": "how many flights, fares or airlines there are",
        "atis_restriction": "restrictions that apply to a fare",
    },
}

CONNECTIVES = {
    "zh": ["，然后", "，另外", "，还有", "，顺便", "，接着", "。再帮我"],
    "en": [" and ", " and then ", " , also ", " and after that ", " , then "],
    "ja": ["、それから", "、あと", "、ついでに"],
    "de": [" und ", " und dann ", " , außerdem "],
    "fr": [" et ", " et ensuite ", " , aussi "],
    "es": [" y ", " y luego ", " , también "],
}


def write_split(out: str, name: str, rows):
    with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_lab = sum(len(r["labels"]) for r in rows) / max(1, len(rows))
    print("  %-5s %6d examples, %.2f labels each" % (name, len(rows), n_lab))


def write_labels(out: str, labels: dict, instructions: str):
    with open(os.path.join(out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"instructions": instructions, "labels": labels}, f, indent=2, ensure_ascii=False)
    print("  labels.json: %d labels" % len(labels))


def parse_slu(text: str):
    """AGIF layout: one 'token TAG' per line, then a line of '#'-joined intents, then a blank."""
    rows, words = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            words = []
        elif len(parts) == 1 and words:
            rows.append({"text": " ".join(words), "labels": sorted(set(parts[0].split("#")))})
            words = []
        else:
            words.append(parts[0])
    return rows


def prepare_slu(name: str, out: str):
    folder = {"mixsnips": "MixSNIPS_clean", "mixatis": "MixATIS_clean"}[name]
    seen = set()
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(AGIF % (folder, split), timeout=60) as r:
            rows = parse_slu(r.read().decode("utf-8"))
        seen.update(l for row in rows for l in row["labels"])
        write_split(out, split, rows)
    desc = DESCRIPTIONS[name]
    write_labels(out, {l: desc.get(l) for l in sorted(seen)},
                 "Which intents does the user express in `utterance`? Several may apply.")


def prepare_massive(out: str, lang: str, sizes: dict, seed: int):
    from huggingface_hub import hf_hub_download

    joiners = CONNECTIVES.get(lang.split("-")[0], [" , "])
    rng = random.Random(seed)
    labels = set()
    for split, src in (("train", "train"), ("dev", "validation"), ("test", "test")):
        path = hf_hub_download("mteb/amazon_massive_intent", "%s/%s.json.gz" % (src, lang), repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            pool = [json.loads(line) for line in f if line.strip()]
        labels.update(r["label_text"] for r in pool)
        rows = []
        while len(rows) < sizes[split]:
            picked, used, n = [], set(), rng.choice([1, 2, 2, 2, 3])
            for r in rng.sample(pool, 8):  # oversample: parts must have distinct intents
                if r["label_text"] not in used and len(picked) < n:
                    picked.append(r)
                    used.add(r["label_text"])
            text = picked[0]["text"]
            for r in picked[1:]:
                text += rng.choice(joiners) + r["text"]
            rows.append({"text": text, "labels": sorted(used)})
        write_split(out, split, rows)
    write_labels(out, {l: l.replace("_", " ") for l in sorted(labels)},
                 "Which intents does the user express in `utterance`? Several may apply.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["mixsnips", "mixatis", "massive"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh-CN", help="massive only: locale, e.g. zh-CN, en, ja-JP")
    ap.add_argument("--n-train", type=int, default=12000, help="massive only")
    ap.add_argument("--n-dev", type=int, default=1500, help="massive only")
    ap.add_argument("--n-test", type=int, default=2000, help="massive only")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("%s -> %s" % (args.dataset, args.out))
    if args.dataset == "massive":
        prepare_massive(args.out, args.lang, {"train": args.n_train, "dev": args.n_dev, "test": args.n_test},
                        args.seed)
    else:
        prepare_slu(args.dataset, args.out)


if __name__ == "__main__":
    main()
