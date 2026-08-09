# 🌾 Agricultural RAG Chatbot

An **AI-powered agricultural advisory chatbot** designed to help farmers with crop selection, fertilizer recommendations, plant disease diagnosis, and agriculture-related questions.

The system combines **Retrieval-Augmented Generation (RAG)** with machine-learning models to provide practical, context-aware agricultural guidance.
## 🚀 Live Demo

🌐 Deployed Application:
https://agri-rag-chatbot.onrender.com/

### Sample Queries

- What is the best fertilizer for rice?
- How do I control leaf blight in paddy?
- Recommend crops for N=90, P=42, K=43, pH=6.5
- Suggest fertilizer plan for tomato cultivation

## 🚀 Key Features

### 📚 1. Agricultural RAG Chatbot

Uses **Retrieval-Augmented Generation (RAG)** to answer agriculture-related questions from trusted reference documents.

**Knowledge sources include:**

* ICAR agricultural guides
* TNAU agricultural resources
* FAO documents
* Wikipedia background information

The documents are processed, chunked, embedded, and stored in **ChromaDB** for semantic retrieval.

**Technology stack:**

* LangChain
* ChromaDB
* Vector embeddings
* Groq LLM
* LCEL

---

### 🌱 2. Crop Recommendation

Recommends suitable crops based on environmental and soil parameters.

**Input features:**

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Soil pH
* Rainfall
* Temperature
* Humidity

A **Random Forest classifier** is trained on crop recommendation data and saved as a serialized model.

**Output:**

* Recommended crop based on the provided conditions.

---

### 🧪 3. Fertilizer Advisor

Provides fertilizer recommendations by calculating nutrient gaps between the current soil condition and the requirements of the selected crop.

The system uses a **rule-based nutrient-gap calculator** to generate recommendations and a dosage calendar.

**Example inputs:**

* Crop
* Current N-P-K values
* Target nutrient requirements

**Output:**

* Nutrient deficiency
* Recommended fertilizer
* Suggested dosage
* Application schedule

---

### 🍃 4. Plant Disease Diagnosis

Provides plant disease diagnosis using either:

* Leaf images
* Text-based symptoms

The planned disease classifier uses **CNN / transfer learning** and is designed around the **PlantVillage dataset**.

The system can provide:

* Possible disease
* Symptoms
* Recommended treatment
* Preventive measures

> **Note:** The current `disease_classifier.py` contains the interface and placeholder inference method. A trained PlantVillage-based transfer-learning model needs to be integrated separately.

---

## 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │      Farmer Input     │
                         │ Text / Image / Data   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Application Router │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌────────────────┐
      │      RAG      │     │ Crop Model    │     │ Disease Model  │
      │   Questions   │     │ Recommendation│     │   Diagnosis    │
      └───────┬───────┘     └───────────────┘     └────────────────┘
              │
              ▼
      ┌───────────────┐
      │   ChromaDB    │
      │ Vector Store   │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Groq LLM     │
      │ Llama / Mixtral│
      └───────┬───────┘
              │
              ▼
      ┌──────────────────────┐
      │ Agricultural Advice  │
      └──────────────────────┘
```

---

## 📁 Project Structure

```text
agri-rag-chatbot/
│
├── README.md
├── requirements.txt
├── .env.example
├── config.py
│
├── data/
│   ├── raw_pdfs/
│   │   └── ICAR_TNAU_FAO_guides/
│   │
│   ├── raw_pdfs_wikipedia/
│   │
│   ├── datasets/
│   │   ├── crop_recommendation.csv
│   │   ├── fertilizer_data.csv
│   │   └── PlantVillage/
│   │
│   └── chroma_db/
│       └── # Generated vector database
│
├── src/
│   ├── ingestion/
│   │   └── ingest.py
│   │
│   ├── rag/
│   │   └── chain.py
│   │
│   ├── models/
│   │   ├── crop_recommender.py
│   │   ├── fertilizer_advisor.py
│   │   └── disease_classifier.py
│   │
│   ├── utils/
│   │   └── logging_config.py
│   │
│   └── app.py
│
├── scripts/
│   ├── run_ingestion.py
│   └── train_crop_model.py
│
├── notebooks/
│   └── train_models.ipynb
│
└── tests/
    └── test_rag_chain.py
```

---

## 🛠️ Technology Stack

| Component                 | Technology                                   |
| ------------------------- | -------------------------------------------- |
| Programming Language      | Python 3.10+                                 |
| LLM                       | Groq                                         |
| LLM Integration           | LangChain Groq                               |
| RAG Framework             | LangChain                                    |
| Vector Database           | ChromaDB                                     |
| Embeddings                | Sentence Transformer / compatible embeddings |
| Crop Recommendation       | Random Forest                                |
| Fertilizer Recommendation | Rule-based system                            |
| Disease Detection         | CNN / Transfer Learning                      |
| Dataset                   | PlantVillage                                 |
| API/UI                    | CLI currently                                |
| Testing                   | Pytest                                       |

---

## 📋 Requirements

* Python **3.10+**
* Groq API key
* Agricultural reference PDFs
* Crop recommendation dataset
* Fertilizer dataset
* PlantVillage dataset for disease classification

Python 3.10+ is recommended because the project uses modern union type syntax such as:

```python
str | None
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd agri-rag-chatbot
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

On Windows, you can also create the `.env` file manually.

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 📚 Preparing the Agricultural Knowledge Base

Place agricultural PDF documents inside:

```text
data/raw_pdfs/
```

For example:

```text
data/raw_pdfs/
├── ICAR/
├── TNAU/
└── FAO/
```

Wikipedia/background documents can be placed inside:

```text
data/raw_pdfs_wikipedia/
```

The ingestion pipeline will:

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
ChromaDB
```

---

## 🔎 Building the Vector Database

Run:

```bash
python scripts/run_ingestion.py
```

This will process the agricultural documents and create the persisted ChromaDB vector store.

The generated database will be stored in:

```text
data/chroma_db/
```

> The ChromaDB directory should generally be excluded from Git because it is generated from the source documents.

---

## 🌱 Training the Crop Recommendation Model

Place the crop recommendation dataset at:

```text
data/datasets/crop_recommendation.csv
```

The dataset should contain features such as:

```text
N
P
K
temperature
humidity
ph
rainfall
label
```

Train the model using:

```bash
python scripts/train_crop_model.py
```

The trained model is then saved for inference.

---

## 💬 Running the Chatbot

After completing ingestion and model training:

```bash
python src/app.py
```

The application currently runs as a **CLI-based chatbot**.

Example:

```text
Farmer: Which crop should I grow if my soil has high nitrogen?

Assistant: Based on the provided soil conditions and agricultural
knowledge base, suitable crops may include...
```

---

## 🧩 RAG Pipeline

The RAG component follows this workflow:

```text
User Question
     ↓
Query Embedding
     ↓
ChromaDB Retriever
     ↓
Relevant Agricultural Documents
     ↓
Context + User Question
     ↓
Prompt
     ↓
Groq LLM
     ↓
Agricultural Answer
```

The retriever uses configurable parameters such as:

* Chunk size
* Chunk overlap
* Number of retrieved documents (`k`)
* Embedding model
* LLM model

These settings are centralized in:

```text
config.py
```

---

## 🤖 Why Groq?

The project uses **Groq** as the LLM backend because it provides fast inference and integrates directly with LangChain through `langchain-groq`.

`ChatGroq` follows LangChain's standard chat-model interface, allowing it to work naturally with the LCEL-based RAG pipeline.

Example model configuration:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
```

This makes the chatbot suitable for interactive applications where low response latency is important.

---

## 🧪 Testing

RAG functionality can be tested using:

```bash
pytest
```

The main test file is:

```text
tests/test_rag_chain.py
```

---

## 🔐 Environment Variables

Create `.env` using `.env.example`.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your actual API key to GitHub.

Recommended `.gitignore` entries:

```text
.env
venv/
__pycache__/
data/chroma_db/
*.pkl
.ipynb_checkpoints/
```

---

## 📊 Example Use Cases

### Crop Recommendation

```text
Input:
N = 90
P = 40
K = 40
pH = 6.5
Rainfall = 200
Temperature = 25

Output:
Recommended crop: ...
```

### Fertilizer Advice

```text
Input:
Crop: Rice
Current NPK: 40-20-20

Output:
Nitrogen deficiency detected.
Recommended fertilizer application:
...
```

### Disease Diagnosis

```text
Input:
"My tomato leaves have yellow spots and brown lesions."

Output:
Possible disease: ...
Recommended treatment: ...
Preventive measures: ...
```

### Agricultural Question Answering

```text
Farmer:
What is the best time to apply nitrogen fertilizer?

Assistant:
[Answer generated using retrieved agricultural documents]
```

---

## 🔮 Future Improvements

The following improvements are planned:

* [ ] Complete PlantVillage disease classification model
* [ ] CNN / transfer-learning based disease detection
* [ ] Image upload support
* [ ] Streamlit web interface
* [ ] FastAPI backend
* [ ] Voice-based farmer interaction
* [ ] Regional/language support
* [ ] Marathi and Hindi support
* [ ] Region-based crop recommendations
* [ ] Crop-specific metadata filtering
* [ ] Weather API integration
* [ ] Soil sensor integration
* [ ] Live agricultural market information
* [ ] Conversation memory
* [ ] Improved RAG evaluation
* [ ] Hallucination detection
* [ ] Retrieval quality evaluation
* [ ] Docker deployment
* [ ] Cloud deployment

---

## ⚠️ Current Limitations

* The chatbot currently uses a **CLI interface**.
* Disease classification inference is currently a placeholder.
* Regional metadata filtering is not yet implemented.
* Crop recommendations depend on the quality and distribution of the training dataset.
* Fertilizer recommendations are rule-based and should be validated against local agricultural guidelines.
* RAG responses depend on the quality and coverage of the supplied agricultural documents.

> **Important:** This system is intended as an AI-assisted agricultural information tool. Fertilizer, pesticide, and disease-treatment recommendations should be verified with qualified agricultural experts and locally applicable guidelines before real-world application.

---

## 👩‍💻 Project Development

This project demonstrates the integration of:

**Generative AI + RAG + Vector Databases + Machine Learning + Computer Vision**

into a practical agricultural advisory system.

### Core AI Components

```text
                    Agricultural AI Assistant
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
        RAG             ML Recommendation    Computer Vision
          │                   │                   │
      LangChain          Random Forest       CNN / Transfer
      ChromaDB                                Learning
      Groq LLM
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    Farmer Advisory System
```

---

