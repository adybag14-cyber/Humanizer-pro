"""Materialize the original synthetic preference corpus used by this pilot."""

import json
from pathlib import Path


def pairs():
    rows = []
    for i in range(4):
        hours = 6 + i * 2
        rows.append({
            "id": f"court-{i}",
            "source": f"People who have reserved a tennis court are asked to cancel their booking at least {hours} hours before it is due to begin if they no longer intend to play. A cancellation made exactly {hours} hours before the booking qualifies for a credit. Later cancellations receive no credit. The credit can be used for another court booking and cannot be exchanged for cash.",
            "chosen": f"Cancel an unused tennis court booking at least {hours} hours before it starts to receive a credit. Cancelling exactly {hours} hours before the start qualifies. Later cancellations receive no credit. The credit covers another court booking and cannot be exchanged for cash.",
            "rejected": f"Cancel an unused tennis court booking more than {hours} hours before it starts to receive a credit. Cancelling exactly {hours} hours before the start does not qualify. Later cancellations receive no credit. The credit covers another court booking and cannot be exchanged for cash.",
        })
        value = 180 + i * 20
        rows.append({
            "id": f"greenhouse-{i}",
            "source": f"The greenhouse team should inspect the soil sensor whenever it records a value below {value}. The inspector must send the reading to the grower, who decides whether watering is needed. An inspector may not start the irrigation system without that decision. If the sensor gives no reading, the inspector should report the missing reading and must not replace it with a zero.",
            "chosen": f"Inspect the soil sensor when its reading falls below {value}. Send the reading to the grower, who decides whether to water. Wait for that decision before starting irrigation. Report an absent reading as missing and never replace it with zero.",
            "rejected": f"Inspect the soil sensor when its reading falls below {value}. Send the reading to the grower, who decides whether to water. Wait for that decision before starting irrigation. Report an absent reading as zero.",
        })
        date = 12 + i
        rows.append({
            "id": f"repairs-{i}",
            "source": f"The repair team expects replacement hinges to arrive on {date} June, but the supplier has not confirmed that date. Customers can leave their damaged equipment at reception in the meantime. Leaving equipment does not reserve a repair slot. The coordinator will contact each customer after the parts have arrived to agree a time for the repair.",
            "chosen": f"Replacement hinges are expected on {date} June, although the supplier has yet to confirm the date. Customers can leave damaged equipment at reception. Doing so does not reserve a repair slot. Once the parts arrive, the coordinator will contact each customer to arrange a repair time.",
            "rejected": f"Replacement hinges will arrive on {date} June. Customers can leave damaged equipment at reception. Doing so does not reserve a repair slot. Once the parts arrive, the coordinator will contact each customer to arrange a repair time.",
        })
        weeks = 3 + i
        rows.append({
            "id": f"subscription-{i}",
            "source": f"Members can pause their meal subscription for up to {weeks} weeks by contacting the membership office. Requests are accepted only before the next payment has been taken. The pause begins with the next unpaid delivery. Any delivery that has already been paid for will go ahead, unless the membership office separately agrees to cancel it.",
            "chosen": f"Contact the membership office before your next payment to pause your meal subscription for up to {weeks} weeks. The pause starts with the next unpaid delivery. Paid deliveries still go ahead unless the office separately agrees to cancel them.",
            "rejected": f"Contact the membership office before your next payment to pause your meal subscription for up to {weeks} weeks. The pause starts with the next delivery, including deliveries you have already paid for. The office will cancel those deliveries.",
        })
        pounds = 35 + i * 5
        rows.append({
            "id": f"travel-{i}",
            "source": f"Volunteers may claim back up to {pounds} pounds for travel to the training day. They must give their receipts to the finance assistant. The training organiser checks that each person attended, but does not approve or pay the claim. The finance manager decides whether to approve a claim once the receipts and attendance confirmation have both been received.",
            "chosen": f"Volunteers can claim up to {pounds} pounds for travel to the training day. Give receipts to the finance assistant. The organiser confirms attendance. The finance manager approves claims after receiving both the receipts and attendance confirmation.",
            "rejected": f"Volunteers can claim up to {pounds} pounds for travel to the training day. Give receipts to the finance assistant. The organiser confirms attendance and approves claims after receiving both the receipts and attendance confirmation.",
        })
        minutes = 18 + i * 3
        rows.append({
            "id": f"ceramics-{i}",
            "source": f"If the glaze demonstration runs more than {minutes} minutes late, the instructor will offer everyone a place at a replacement session. A delay of exactly {minutes} minutes does not trigger this offer. Students can choose whether to accept the replacement place. There is no additional charge, and the original session will continue even if some students choose a replacement.",
            "chosen": f"The instructor will offer everyone a free replacement session if the glaze demonstration is more than {minutes} minutes late. A delay of exactly {minutes} minutes does not qualify. Students can accept or decline, and the original session continues.",
            "rejected": f"The instructor will offer everyone a free replacement session if the glaze demonstration is at least {minutes} minutes late. A delay of exactly {minutes} minutes qualifies. Students can accept or decline, and the original session continues.",
        })
        count = 8 + i
        rows.append({
            "id": f"bicycle-{i}",
            "source": f"The bicycle workshop has {count} spare helmets available for people attending its maintenance class. A participant who asks for a helmet before the class begins can borrow one while supplies last. Helmet requests are handled in the order received. The workshop has not stated whether more helmets will become available later, so staff should not promise additional stock.",
            "chosen": f"The maintenance class has {count} spare helmets to lend. Ask before the class begins. Requests are handled in order while helmets remain. Staff should make no promise of more stock because the workshop has given no information about future availability.",
            "rejected": f"The maintenance class has {count} spare helmets to lend. Ask before the class begins. Requests are handled in order while helmets remain. Staff should tell participants that more helmets will arrive later.",
        })
        total = 40 + i * 5
        rows.append({
            "id": f"garden-{i}",
            "source": f"The garden club has space for {total} people at the seed exchange. Members can bring one guest each, but guests count towards the same capacity limit. The secretary will stop accepting reservations when that limit is reached. A member who reserves a place and later cannot attend should tell the secretary so that the place can be offered to someone else.",
            "chosen": f"The seed exchange can hold {total} people, including guests. Each member may bring one guest. The secretary will close reservations at capacity. Tell the secretary if you can no longer attend so someone else can take your place.",
            "rejected": f"The seed exchange can hold {total} members, plus their guests. Each member may bring one guest. The secretary will close reservations at capacity. Tell the secretary if you can no longer attend so someone else can take your place.",
        })
    return rows


if __name__ == "__main__":
    path = Path(__file__).with_name("pairs.json")
    path.write_text(json.dumps(pairs(), indent=2) + "\n", encoding="utf-8")
