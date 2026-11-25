"""Rank complete semantic reviews using the frozen lexicographic protocol."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUSES = {"preserved", "omitted", "contradicted", "uncertain"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(directory, candidate, fixtures):
    outputs = read_json(directory / (candidate + ".outputs.json"))
    review = read_json(directory / (candidate + ".review.json"))
    if review["candidate"] != candidate or outputs["candidate"] != candidate:
        raise ValueError("Candidate label mismatch")
    expected = [case["id"] for case in fixtures]
    if [case["case"] for case in review["cases"]] != expected or [case["case"] for case in outputs["cases"]] != expected:
        raise ValueError("Review cases do not match the frozen fixtures")
    acceptable = errors = edited = clarity = preserved = 0
    details = []
    for fixture, generated, assessed in zip(fixtures, outputs["cases"], review["cases"]):
        claims = assessed["claims"]
        complete = (
            assessed["review_complete"] is True
            and [claim["claim"] for claim in claims] == fixture["claims"]
            and all(claim["status"] in STATUSES for claim in claims)
            and isinstance(assessed["unsupported_assertions"], list)
            and all(isinstance(item, str) and item.strip() for item in assessed["unsupported_assertions"])
            and type(assessed["clarity"]) is int and assessed["clarity"] in (0, 1, 2)
            and type(assessed["editing_goal_met"]) is bool
        )
        if not complete:
            raise ValueError(candidate + " needs a complete review for " + fixture["id"])
        if any(claim["status"] != "preserved" and not claim.get("evidence", "").strip() for claim in claims):
            raise ValueError(candidate + " needs evidence for every failed or uncertain claim")
        if fixture["edit_required"] and generated["copy_after_whitespace_normalization"] and assessed["editing_goal_met"]:
            raise ValueError("An unchanged output cannot meet a required editing goal")
        bad = sum(claim["status"] != "preserved" for claim in claims) + len(assessed["unsupported_assertions"])
        good = (
            generated["output"]["success"] and bad == 0
            and assessed["clarity"] >= 1 and assessed["editing_goal_met"]
        )
        acceptable += int(good)
        errors += bad
        preserved += sum(claim["status"] == "preserved" for claim in claims)
        edited += int(fixture["edit_required"] and assessed["editing_goal_met"])
        clarity += assessed["clarity"]
        details.append({"case": fixture["id"], "acceptable": bool(good), "fidelity_failures": bad})
    return {
        "candidate": candidate, "fully_acceptable_cases": acceptable, "fidelity_failures": errors,
        "preserved_claims": preserved, "required_edit_goals_met": edited, "clarity_sum": clarity,
        "ranking_key": [acceptable, -errors, edited, clarity], "cases": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.directory / "manifest.json")
    fixtures = read_json(HERE / "cases.json")
    rows = [score(args.directory, candidate["id"], fixtures) for candidate in manifest["candidates"]]
    rows.sort(key=lambda row: row["ranking_key"], reverse=True)
    prior = None
    for position, row in enumerate(rows, 1):
        if row["ranking_key"] != prior:
            rank = position
        row["rank"] = rank
        prior = row["ranking_key"]
    print(json.dumps({"case_count": len(fixtures), "ranked": rows}, indent=2))


if __name__ == "__main__":
    main()
