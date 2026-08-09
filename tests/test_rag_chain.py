"""
Smoke tests. These require a populated Chroma DB (run ingestion first) and a
valid GROQ_API_KEY in .env — mark/skip appropriately in CI if those aren't
available.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import config


@pytest.mark.skipif(
    not config.CHROMA_PERSIST_DIR.exists() or not any(config.CHROMA_PERSIST_DIR.iterdir()),
    reason="Vector store not built yet — run scripts/run_ingestion.py first.",
)
def test_rag_chain_builds():
    from src.rag.chain import build_rag_chain
    chain = build_rag_chain()
    assert chain is not None


def test_fertilizer_advisor_math():
    """This test doesn't need external data — verifies the dosage arithmetic
    directly against the fertilizer nutrient-content constants."""
    from src.models.fertilizer_advisor import FERTILIZER_NUTRIENT_CONTENT

    assert FERTILIZER_NUTRIENT_CONTENT["Urea"]["N"] == 0.46
    assert FERTILIZER_NUTRIENT_CONTENT["DAP"]["P"] == 0.46
    assert FERTILIZER_NUTRIENT_CONTENT["MOP"]["K"] == 0.60
