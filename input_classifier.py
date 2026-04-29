import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a clinical query classifier, a medical decision-support system. Your only job is to detect whether an incoming query is a pure guideline retrieval question with no patient-specific context, or whether it contains any patient-specific information that requires clinical reasoning.

You output JSON only. No preamble, no explanation, no markdown fences.

DEFINITIONS
-----------
GUIDELINE_ONLY: The query asks what a guideline says, what current recommendations are, or what the evidence states — with no specific patient described. No age, sex, comorbidities, lab values, symptoms, or clinical context are present.

PATIENT_SPECIFIC: The query contains any patient descriptor — age, sex, presenting symptom, comorbidity, lab value, medication, clinical setting, or implied urgency. Even a single patient detail makes this PATIENT_SPECIFIC.


CLASSIFICATION RULES
--------------------
1. If in doubt, output PATIENT_SPECIFIC. The cost of misclassifying a patient query as guideline-only is higher than the reverse.
2. A query mentioning a disease or drug name WITHOUT a patient is GUIDELINE_ONLY.
3. A query mentioning a disease or drug WITH any patient descriptor is PATIENT_SPECIFIC.
4. Rhetorical patient framing ("for a typical patient with X") is PATIENT_SPECIFIC.
5. Questions about dosing WITHOUT a specific patient are GUIDELINE_ONLY.
6. Questions about dosing WITH a specific patient profile are PATIENT_SPECIFIC."""

USER_PROMPT_TEMPLATE = """NOW CLASSIFY THIS QUERY:
Query: "{query_text}"

Respond with JSON in exactly this format:
{{
  "classification": "<GUIDELINE_ONLY|PATIENT_SPECIFIC>",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "reasoning": "<one sentence explaining the key signal>"
}}"""


def classify(query_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query_text=query_text)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def classify_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        records = json.load(f)

    results = []
    for record in records:
        query = record["input"]["question"]
        classification = classify(query)
        results.append({
            "id": record["id"],
            "question": query,
            **classification,
        })
        print(f"[{record['id']}] {classification['classification']} ({classification['confidence']}) — {query[:60]}...")

    return results


if __name__ == "__main__":
    results = classify_dataset("vera_answers_extras.json")
    print("\n=== Classification Summary ===")
    for r in results:
        print(f"{r['classification']:20s} | {r['confidence']:6s} | {r['question'][:70]}")
