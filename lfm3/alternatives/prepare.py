"""Create the fixed, independently authored factual-editing training sets."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    weighted = []
    unlikelihood = []
    for i in range(4):
        day = ["Monday", "Wednesday", "Saturday", "Sunday"][i]
        hour = ["08:15", "09:25", "14:35", "16:45"][i]
        actor = ["caretaker", "receptionist", "coordinator", "site manager"][i]
        room = ["north", "south", "east", "west"][i]
        amount = 35 + i * 7
        period = 3 + i
        weighted.extend([
            {
                "source": f"It is necessary for the {actor} to unlock the {room} entrance at {hour} on {day}. The instruction applies on {day} only.",
                "target": f"The {actor} must unlock the {room} entrance at {hour} on {day}. This applies on {day} only.",
                "critical": [actor, room, hour, day, "only"],
            },
            {
                "source": f"The permit gives permission for {amount} chairs to be placed in the courtyard. It does not give permission for chairs to be placed on the pavement.",
                "target": f"The permit allows {amount} chairs in the courtyard. It does not allow chairs on the pavement.",
                "critical": [str(amount), "courtyard", "does not", "pavement"],
            },
            {
                "source": f"The repairs might be finished by {day}. This is an estimate and it has not been confirmed by the contractor.",
                "target": f"The repairs might finish by {day}. The contractor has not confirmed this estimate.",
                "critical": ["might", day, "has not confirmed", "estimate"],
            },
            {
                "source": f"The {actor} is responsible for booking the room. The chair of the meeting is responsible for sending the agenda {period} days before the meeting.",
                "target": f"The {actor} books the room. The meeting chair sends the agenda {period} days before the meeting.",
                "critical": [actor, "books the room", "meeting chair", str(period), "before"],
            },
            {
                "source": f"A charge of {amount} pounds applies to each replacement key. The charge is waived if the original key was faulty when it was issued.",
                "target": f"Each replacement key costs {amount} pounds. There is no charge if the original key was faulty when issued.",
                "critical": [str(amount), "Each", "no charge", "if", "faulty when issued"],
            },
            {
                "source": f"The register contains {amount} names. The number of people who actually attended is unknown, so the register must not be described as an attendance count.",
                "target": f"There are {amount} names on the register. Actual attendance is unknown, so the register must not be called an attendance count.",
                "critical": [str(amount), "names", "unknown", "must not", "attendance count"],
            },
        ])
        count = 6 + 2 * i
        minutes = 17 + i * 3
        months = 7 + i
        actor2 = ["editor", "archivist", "organiser", "registrar"][i]
        site = ["Hill", "River", "Park", "Valley"][i]
        unlikelihood.extend([
            {
                "source": f"The {site} club can lend no more than {count} maps to one group. The {actor2} must approve the loan.",
                "target": f"The {site} club can lend a group up to {count} maps. The {actor2} must approve the loan.",
                "rejected": f"The {site} club can lend a group at least {count} maps. The {actor2} must approve the loan.",
                "rejected_spans": ["at least"],
            },
            {
                "source": f"The interview starts {minutes} minutes after the briefing ends. Participants are asked to remain in the lobby until called.",
                "target": f"The interview starts {minutes} minutes after the briefing ends. Participants should wait in the lobby until called.",
                "rejected": f"The interview starts {minutes} minutes before the briefing ends. Participants should wait in the lobby until called.",
                "rejected_spans": ["before"],
            },
            {
                "source": f"The {actor2} must obtain the author's permission before publishing the photograph. The designer is responsible only for the layout.",
                "target": f"The {actor2} must get the author's permission before publishing the photograph. The designer handles only the layout.",
                "rejected": f"The designer must get the author's permission before publishing the photograph. The designer handles only the layout.",
                "rejected_spans": ["designer"],
            },
            {
                "source": f"The vouchers expire after {months} months. They can be used at the {site} branch only.",
                "target": f"The vouchers expire after {months} months and can be used only at the {site} branch.",
                "rejected": f"The vouchers expire after {months + 1} months and can be used only at the {site} branch.",
                "rejected_spans": [str(months + 1)],
            },
            {
                "source": f"The {site} committee has not decided whether to extend the programme. A decision may be made on {day}, but this has not been promised.",
                "target": f"The {site} committee has yet to decide whether to extend the programme. It may decide on {day}, but has made no promise.",
                "rejected": f"The {site} committee has decided to extend the programme. It may decide on {day}, but has made no promise.",
                "rejected_spans": ["has decided to extend"],
            },
            {
                "source": f"The award is open to applicants with at least {count} months of experience. Holding a qualification is optional.",
                "target": f"Applicants need at least {count} months of experience for the award. A qualification is optional.",
                "rejected": f"Applicants need at least {count} months of experience for the award. A qualification is required.",
                "rejected_spans": ["required"],
            },
        ])
    for name, rows in [("weighted", weighted), ("unlikelihood", unlikelihood)]:
        (ROOT / f"{name}-pairs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main()
