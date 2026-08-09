"""
Fertilizer & Nutrient Management Advisor.

Rule-based (not ML) — compares current soil N-P-K against a target crop's
requirement table and converts the gap into approximate fertilizer dosage
(Urea / DAP / MOP, kg/acre). This is intentionally simple/transparent since
farmers need to trust and verify the numbers; RAG then supplements this with
application-timing guidance (basal vs split doses) pulled from ChromaDB.

Expected CSV at: data/datasets/crop_nutrient_requirements.csv
    columns: crop, N_required, P_required, K_required   (kg/acre)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import config

REQUIREMENTS_CSV = config.DATASETS_DIR / "crop_nutrient_requirements.csv"

# Nutrient content fraction of common fertilizers, used to convert a raw
# nutrient deficit (kg/acre of N, P2O5, K2O) into a bag-able fertilizer amount.
FERTILIZER_NUTRIENT_CONTENT = {
    "Urea": {"N": 0.46},
    "DAP": {"N": 0.18, "P": 0.46},
    "MOP": {"K": 0.60},
}


def _load_requirements() -> pd.DataFrame:
    if not REQUIREMENTS_CSV.exists():
        raise FileNotFoundError(
            f"{REQUIREMENTS_CSV} not found. Add a crop nutrient requirements "
            "table with columns: crop, N_required, P_required, K_required."
        )
    return pd.read_csv(REQUIREMENTS_CSV)


def compute_deficit(current: dict, crop: str) -> dict:
    """
    current: {"N": .., "P": .., "K": ..}  (kg/acre currently in soil)
    Returns nutrient deficits (kg/acre), clipped at 0.
    """
    df = _load_requirements()
    row = df[df["crop"].str.lower() == crop.lower()]
    if row.empty:
        raise ValueError(f"No nutrient requirement data for crop '{crop}'")
    row = row.iloc[0]

    deficit = {
        "N": max(row["N_required"] - current.get("N", 0), 0),
        "P": max(row["P_required"] - current.get("P", 0), 0),
        "K": max(row["K_required"] - current.get("K", 0), 0),
    }
    return deficit


def recommend_fertilizer_dosage(current: dict, crop: str) -> dict:
    """
    Converts nutrient deficit into an approximate fertilizer bag plan.
    This is a simplified single-source-per-nutrient approach:
    P via DAP (which also contributes some N), remaining N via Urea, K via MOP.
    A real deployment should let agronomists tune this logic per region.
    """
    deficit = compute_deficit(current, crop)

    dap_kg = deficit["P"] / FERTILIZER_NUTRIENT_CONTENT["DAP"]["P"] if deficit["P"] > 0 else 0
    n_from_dap = dap_kg * FERTILIZER_NUTRIENT_CONTENT["DAP"]["N"]
    remaining_n = max(deficit["N"] - n_from_dap, 0)
    urea_kg = remaining_n / FERTILIZER_NUTRIENT_CONTENT["Urea"]["N"] if remaining_n > 0 else 0
    mop_kg = deficit["K"] / FERTILIZER_NUTRIENT_CONTENT["MOP"]["K"] if deficit["K"] > 0 else 0

    return {
        "crop": crop,
        "nutrient_deficit_kg_per_acre": deficit,
        "fertilizer_plan_kg_per_acre": {
            "DAP": round(dap_kg, 1),
            "Urea": round(urea_kg, 1),
            "MOP": round(mop_kg, 1),
        },
    }


if __name__ == "__main__":
    example = recommend_fertilizer_dosage({"N": 40, "P": 15, "K": 20}, crop="Rice")
    print(example)
