"""
Crop Disease Diagnostic module.

Interface for image-based disease classification (trained on PlantVillage:
54k+ leaf images, 14 crops, 38 disease classes). This file defines the
expected API and a transfer-learning training scaffold; a trained weights
file (artifacts/disease_classifier.pt) is required for real inference.

NOT YET TRAINED — predict() will raise until a weights file is provided.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import config

WEIGHTS_PATH = config.MODELS_DIR / "disease_classifier.pt"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    """MobileNetV3-Small transfer-learning backbone — lightweight enough for
    CPU inference, good fit for a farmer-facing tool without GPU serving."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_class_names() -> list[str]:
    """Loaded from a text file saved alongside the weights during training,
    one class name per line (e.g. 'Tomato___Early_blight')."""
    names_path = config.MODELS_DIR / "disease_classes.txt"
    if not names_path.exists():
        raise FileNotFoundError(f"{names_path} not found — train the model first.")
    return names_path.read_text().splitlines()


def predict(image_path: str, top_k: int = 3) -> list[tuple[str, float]]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No trained weights at {WEIGHTS_PATH}. This module is a scaffold — "
            "train on PlantVillage (see notebooks/train_models.ipynb) before use."
        )

    class_names = load_class_names()
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    ranked = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def predict_from_symptoms(symptom_text: str) -> str:
    """
    Fallback path when the farmer types symptoms instead of uploading a photo.
    Delegates to the RAG chain (see src/rag/chain.py) rather than the vision
    model, since this is a text-understanding task, not image classification.
    """
    from src.rag.chain import build_rag_chain
    chain = build_rag_chain()
    return chain.invoke(
        f"A farmer describes these crop symptoms: '{symptom_text}'. "
        "What disease is this most likely to be, and what is the recommended treatment?"
    )
