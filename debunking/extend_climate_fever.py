"""
Extend the Climate FEVER evidence base with two UK specific claims.

The evidence base released with the debunker has no entries for the
geoengineering and "chemtrail" conspiracy claims that recur in UK discourse.
Two claims are added, each with evidence taken from statements the Met Office
has published.

Updates debunking/climate_fever_cards.csv in place. Safe to run more than once.
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "climate_fever_cards.csv"

df = pd.read_csv(CSV_PATH)

new = pd.DataFrame([
    {"claim_id": 3135,
     "claim": "The Met Office is covertly using geoengineering to control the climate",
     "claim_label": 1,
     "evidences": "[{'evidence': 'The Met Office does not advocate or provide context for real-world physical experiments, for example that adds chemicals to the atmosphere, nor does it partake in weather modification.'}, {'evidence': 'Any research the Met Office conducts in this field is entirely confined to computer models being run in a virtual world.'}, {'evidence': 'The Met Office contribution to scientific research in the field of geoengineering is based on simulations using climate models.'}]",
     "CARDS_label": "5_3 Climate change related dilemmas are conspiracies"},
    {"claim_id": 3136,
     "claim": "Aircraft trails are chemicals sprayed in a secret weather modification plan that is causing climate change",
     "claim_label": 1,
     "evidences": "[{'evidence': 'Contrails, or condensation trails, are formed by mixing between aircraft engine exhaust air and the surrounding natural air. The exhaust air contains additional water vapour that has been released as a by-product of burning fuel.'}, {'evidence': 'Above 20,000 feet, the cruising altitude of most commercial airliners, the air surrounding the aircraft is well below freezing, causing the water vapour to condense rapidly and then freeze. Tiny particles from the engine, known as condensation nuclei, act as a starting point for condensation, leaving thin trails of ice crystals behind the aircraft engines.'}, {'evidence': 'If the air is very dry, the ice crystals will sublime and become invisible. However, if the air is humid, the water droplets or ice crystals will stay where they are, often spreading out and leaving a fluffy trail where the aircraft has passed. Trails may last for many hours.'}]",
     "CARDS_label": "5_3 Climate change related dilemmas are conspiracies"},
])

# Skip claims that are already present so that a rerun does not duplicate them.
new = new[~new["claim_id"].isin(df["claim_id"])]

pd.concat([df, new]).to_csv(CSV_PATH, index=False)
print(f"{len(new)} claims added")
