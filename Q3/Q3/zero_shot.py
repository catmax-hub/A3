!pip install -q transformers torch

import json
from transformers import pipeline

# 1. Initialize Hugging Face zero-shot pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sample_messages = [
    "There is a water leak in Building C that has been running all morning.",
    "I want to report that one of our suppliers may not meet our sustainability policy.",
    "The accessible entrance near the main building has been blocked for two days."
]

candidate_categories = [
    "Environmental - Facilities",
    "Environmental - Sustainability",
    "Social - Accessibility",
    "Social - Health & Safety",
    "Governance - Procurement Compliance",
    "Non-ESG / General Operations"
]

candidate_urgencies = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW"
]

# 2. Run inference and assemble results into JSON
hf_json_results = []

for msg in sample_messages:
    cat_pred = classifier(msg, candidate_categories, multi_label=False)
    urg_pred = classifier(msg, candidate_urgencies, multi_label=False)

    record = {
        "input_message": msg,
         # probability 0 - 1 category_confidencee and urgency confidence
         # top ranked issue category & urgency
        "issue_category": cat_pred["labels"][0],
        "category_confidence": round(float(cat_pred["scores"][0]), 4),
        "urgency": urg_pred["labels"][0],
        "urgency_confidence": round(float(urg_pred["scores"][0]), 4),
        "followup_required": True if urg_pred["labels"][0] in ["HIGH", "CRITICAL"] else False,
        "followup_action": None,
        "escalation_reason": None,
        "brief_summary": None
    }
    hf_json_results.append(record)

# 3. Print the entire list as formatted JSON
print(json.dumps(hf_json_results, indent=2))
