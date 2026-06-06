"""
LLM Prompt Injection Detector
-------------------------------
Applies unsupervised anomaly detection (Isolation Forest) to detect
prompt injection attempts by modelling the behavioral distribution of
benign prompts and flagging statistical outliers.

Author: Judah Idowu
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import joblib
import json
import re
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class PromptFeatureExtractor:
    """
    Extracts behavioral/statistical features from raw prompt strings.

    Injection attacks tend to exhibit distinctive lexical and structural
    signatures compared to benign user queries:
      - Elevated instruction-override vocabulary
      - Unusual punctuation density
      - Role-assumption patterns
      - Boundary-breaking tokens (delimiters, code fences)
      - Abnormal length distributions
    """

    OVERRIDE_KEYWORDS = [
        "ignore", "disregard", "forget", "override", "bypass",
        "pretend", "act as", "you are now", "new instructions",
        "system prompt", "jailbreak", "do anything now", "dan",
        "ignore previous", "ignore above", "ignore all", "from now on",
        "your new role", "your true self", "without restrictions",
        "no restrictions", "no limits", "unrestricted", "developer mode",
        "sudo", "admin mode", "god mode", "maintenance mode",
        "hypothetically", "theoretically", "in a fictional world"
    ]

    DELIMITERS = [
        "###", "---", "===", "```", "<<<", ">>>",
        "[INST]", "[/INST]", "<s>", "</s>",
        "<system>", "</system>", "<human>", "</human>",
        "<<SYS>>", "<</SYS>>", "SYSTEM:", "USER:", "ASSISTANT:"
    ]

    def extract(self, prompt: str) -> dict:
        p = prompt.lower()
        tokens = p.split()
        sentences = re.split(r'[.!?]+', p)

        features = {}

        # --- Length features ---
        features["char_length"] = len(prompt)
        features["word_count"] = len(tokens)
        features["sentence_count"] = max(len([s for s in sentences if s.strip()]), 1)
        features["avg_word_length"] = np.mean([len(t) for t in tokens]) if tokens else 0
        features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]

        # --- Override vocabulary ---
        features["override_keyword_count"] = sum(
            1 for kw in self.OVERRIDE_KEYWORDS if kw in p
        )
        features["override_keyword_density"] = (
            features["override_keyword_count"] / max(features["word_count"], 1)
        )

        # --- Delimiter / structural injection markers ---
        features["delimiter_count"] = sum(
            1 for d in self.DELIMITERS if d.lower() in p
        )

        # --- Punctuation features ---
        features["exclamation_count"] = prompt.count("!")
        features["question_count"] = prompt.count("?")
        features["colon_count"] = prompt.count(":")
        features["quote_count"] = prompt.count('"') + prompt.count("'")
        features["bracket_count"] = (
            prompt.count("[") + prompt.count("]") +
            prompt.count("{") + prompt.count("}")
        )
        features["punctuation_density"] = sum(
            1 for c in prompt if not c.isalnum() and not c.isspace()
        ) / max(len(prompt), 1)

        # --- Casing features ---
        features["uppercase_ratio"] = (
            sum(1 for c in prompt if c.isupper()) / max(len(prompt), 1)
        )
        features["capitalized_word_ratio"] = (
            sum(1 for t in prompt.split() if t and t[0].isupper()) /
            max(len(prompt.split()), 1)
        )

        # --- Lexical diversity ---
        unique_tokens = set(tokens)
        features["lexical_diversity"] = len(unique_tokens) / max(len(tokens), 1)

        # --- Imperative / command structure ---
        imperative_starts = [
            "ignore", "forget", "stop", "do not", "don't", "never",
            "always", "you must", "you should", "you will", "you shall",
            "print", "repeat", "say", "write", "output", "respond"
        ]
        features["starts_with_imperative"] = int(
            any(p.strip().startswith(imp) for imp in imperative_starts)
        )

        # --- Roleplay / persona injection ---
        persona_patterns = [
            r'\byou are\b', r'\byou\'re\b.*\b(now|actually|really|truly)\b',
            r'\bact as\b', r'\bpretend\b', r'\brole.?play\b',
            r'\bpersona\b', r'\bcharacter\b.*\bwho\b'
        ]
        features["persona_pattern_count"] = sum(
            1 for pat in persona_patterns if re.search(pat, p)
        )

        # --- Nested instruction signals ---
        features["instruction_nesting_depth"] = max(
            p.count("instruction"), p.count("prompt"), p.count("context")
        )

        return features

    def transform(self, prompts: list[str]) -> pd.DataFrame:
        records = [self.extract(p) for p in prompts]
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PromptInjectionDetector:
    """
    Unsupervised prompt injection detector based on Isolation Forest.

    Trained exclusively on benign prompts to model the distribution of
    normal user behaviour. Injection attempts are detected as statistical
    outliers — no labelled attack examples required during training.

    This mirrors the UEBA-style approach from:
      Idowu J. (2025) "Evaluating the Robustness of Isolation Forest in UEBA"
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.extractor = PromptFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, benign_prompts: list[str]) -> "PromptInjectionDetector":
        """Train on benign prompts only."""
        X = self.extractor.transform(benign_prompts)
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print(f"[Detector] Trained on {len(benign_prompts)} benign prompts.")
        print(f"[Detector] Features: {len(self.feature_names)}")
        return self

    def score(self, prompts: list[str]) -> np.ndarray:
        """
        Return anomaly scores. More negative = more anomalous.
        Scores are inverted and normalised to [0, 1] for readability
        (higher score = more likely to be injection).
        """
        assert self.is_fitted, "Call fit() first."
        X = self.extractor.transform(prompts)
        X_scaled = self.scaler.transform(X[self.feature_names])
        raw_scores = self.model.score_samples(X_scaled)
        # Invert and normalise: higher = more anomalous
        inv = -raw_scores
        normalised = (inv - inv.min()) / (inv.max() - inv.min() + 1e-10)
        return normalised

    def predict(self, prompts: list[str], threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 = injection, 0 = benign."""
        scores = self.score(prompts)
        return (scores >= threshold).astype(int)

    def evaluate(
        self,
        prompts: list[str],
        labels: list[int],
        threshold: float = 0.5
    ) -> dict:
        """
        Evaluate against labelled data. Labels: 1 = injection, 0 = benign.
        Returns a dict of metrics including AUC-ROC and Average Precision.
        """
        scores = self.score(prompts)
        preds = (scores >= threshold).astype(int)

        report = classification_report(labels, preds, output_dict=True)
        auc_roc = roc_auc_score(labels, scores)
        avg_precision = average_precision_score(labels, scores)

        results = {
            "threshold": threshold,
            "auc_roc": round(auc_roc, 4),
            "average_precision": round(avg_precision, 4),
            "precision_injection": round(report["1"]["precision"], 4),
            "recall_injection": round(report["1"]["recall"], 4),
            "f1_injection": round(report["1"]["f1-score"], 4),
            "precision_benign": round(report["0"]["precision"], 4),
            "recall_benign": round(report["0"]["recall"], 4),
            "accuracy": round(report["accuracy"], 4),
        }
        return results

    def explain(self, prompt: str) -> dict:
        """Return feature values and anomaly score for a single prompt."""
        features = self.extractor.extract(prompt)
        score = float(self.score([prompt])[0])
        label = "INJECTION" if score >= 0.5 else "BENIGN"
        top_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return {
            "prompt": prompt[:120] + "..." if len(prompt) > 120 else prompt,
            "anomaly_score": round(score, 4),
            "label": label,
            "top_features": dict(top_features),
            "all_features": features
        }

    def save(self, path: str):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators
        }, path)
        print(f"[Detector] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PromptInjectionDetector":
        data = joblib.load(path)
        det = cls(
            contamination=data["contamination"],
            n_estimators=data["n_estimators"]
        )
        det.model = data["model"]
        det.scaler = data["scaler"]
        det.feature_names = data["feature_names"]
        det.is_fitted = True
        return det
