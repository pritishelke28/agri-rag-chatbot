"""
Orchestrator / CLI entry point.

Routes a farmer's request to the right module:
  - "crop"       -> crop_recommender (needs N,P,K,temperature,humidity,ph,rainfall)
  - "fertilizer" -> fertilizer_advisor (needs current N,P,K + target crop)
  - "disease"    -> disease_classifier (image path) or free-text symptom RAG
  - anything else -> general RAG advisory chain

This is intentionally a simple CLI loop so the architecture is easy to follow;
swap in Streamlit/FastAPI for a real UI once the modules are validated.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.chain import build_rag_chain
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MENU = """
=== Agricultural Advisory Chatbot ===
1. Ask a general question (RAG advisory)
2. Crop recommendation (soil N-P-K, pH, rainfall)
3. Fertilizer dosage plan
4. Disease diagnosis from symptoms (text)
q. Quit
> """


def run_general_query(chain):
    question = input("Your question: ").strip()
    print("\n" + chain.invoke(question) + "\n")


def run_crop_recommendation():
    from src.models.crop_recommender import predict_top_k
    try:
        soil_inputs = {
            "N": float(input("N (kg/ha): ")),
            "P": float(input("P (kg/ha): ")),
            "K": float(input("K (kg/ha): ")),
            "temperature": float(input("Temperature (C): ")),
            "humidity": float(input("Humidity (%): ")),
            "ph": float(input("Soil pH: ")),
            "rainfall": float(input("Rainfall (mm): ")),
        }
        results = predict_top_k(soil_inputs, k=3)
        print("\nTop crop recommendations:")
        for crop, prob in results:
            print(f"  {crop}: {prob:.1%} confidence")
        print()
    except FileNotFoundError as e:
        print(f"\n[Model not trained yet] {e}\n")


def run_fertilizer_advisor():
    from src.models.fertilizer_advisor import recommend_fertilizer_dosage
    try:
        crop = input("Target crop: ").strip()
        current = {
            "N": float(input("Current soil N (kg/acre): ")),
            "P": float(input("Current soil P (kg/acre): ")),
            "K": float(input("Current soil K (kg/acre): ")),
        }
        plan = recommend_fertilizer_dosage(current, crop)
        print(f"\nFertilizer plan for {crop}: {plan['fertilizer_plan_kg_per_acre']}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[Data not available] {e}\n")


def run_disease_diagnosis(chain):
    symptoms = input("Describe the symptoms: ").strip()
    from src.models.disease_classifier import predict_from_symptoms
    print("\n" + predict_from_symptoms(symptoms) + "\n")


def main():
    logger.info("Starting Agricultural Advisory Chatbot")
    chain = None
    try:
        chain = build_rag_chain()
    except Exception as e:
        logger.warning(f"RAG chain unavailable (has ingestion been run?): {e}")

    while True:
        choice = input(MENU).strip().lower()
        
        if choice == "q":
            break
        elif choice == "1":
            if chain:
                run_general_query(chain)
            else:
                print("RAG chain not ready — run scripts/run_ingestion.py first.\n")
        elif choice == "2":
            run_crop_recommendation()
        elif choice == "3":
            run_fertilizer_advisor()
        elif choice == "4":
            if chain:
                run_disease_diagnosis(chain)
            else:
                print("RAG chain not ready — run scripts/run_ingestion.py first.\n")
        else:
            print("Invalid option.\n")


if __name__ == "__main__":
    main()
