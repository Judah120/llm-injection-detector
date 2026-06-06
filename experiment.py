"""
Experiment Runner
-----------------
Trains the detector, evaluates across multiple thresholds,
generates evasion curve data, and saves all results to /results.

Run: python src/experiment.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve

from dataset import build_dataset, BENIGN_PROMPTS, INJECTION_PROMPTS
from detector import PromptInjectionDetector, PromptFeatureExtractor


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_main_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline Detection Performance")
    print("="*60)

    # Build dataset
    dataset = build_dataset(output_path="data/dataset.json")

    train_benign = dataset["train_benign_only"]
    test_data = dataset["test"]
    test_prompts = [d["prompt"] for d in test_data]
    test_labels = [d["label"] for d in test_data]

    # Train detector
    detector = PromptInjectionDetector(contamination=0.05, n_estimators=200)
    detector.fit(train_benign)

    # Score all test prompts
    scores = detector.score(test_prompts)

    # Evaluate at multiple thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = []
    for t in thresholds:
        result = detector.evaluate(test_prompts, test_labels, threshold=t)
        threshold_results.append(result)
        print(f"\nThreshold={t:.1f} | "
              f"AUC-ROC={result['auc_roc']:.4f} | "
              f"F1(injection)={result['f1_injection']:.4f} | "
              f"Recall={result['recall_injection']:.4f} | "
              f"Precision={result['precision_injection']:.4f}")

    # Save threshold sweep
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(threshold_results, f, indent=2)

    # PR curve data
    precision, recall, pr_thresholds = precision_recall_curve(test_labels, scores)
    fpr, tpr, roc_thresholds = roc_curve(test_labels, scores)

    pr_curve_data = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist()
    }
    roc_curve_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist()
    }

    with open(RESULTS_DIR / "pr_curve.json", "w") as f:
        json.dump(pr_curve_data, f, indent=2)
    with open(RESULTS_DIR / "roc_curve.json", "w") as f:
        json.dump(roc_curve_data, f, indent=2)

    # Save model
    Path("models").mkdir(exist_ok=True)
    detector.save("models/detector.pkl")

    return detector, scores, test_labels


def run_evasion_experiment(detector: PromptInjectionDetector):
    """
    Security Decay analysis: systematically dilute injection prompts
    by mixing them with benign content at varying ratios.
    Measures how detection degrades as attack stealth increases.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Evasion / Security Decay Analysis")
    print("="*60)

    injection_samples = INJECTION_PROMPTS[:20]
    benign_filler = "Can you help me understand how machine learning works? "

    dilution_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    evasion_results = []

    for dilution in dilution_levels:
        diluted_prompts = []
        for inj in injection_samples:
            # Prepend benign content proportional to dilution level
            filler_words = int(dilution * len(inj.split()))
            filler = " ".join([benign_filler.split()[i % len(benign_filler.split())]
                               for i in range(filler_words)])
            diluted = (filler + " " + inj).strip() if filler else inj
            diluted_prompts.append(diluted)

        scores = detector.score(diluted_prompts)
        detection_rate = float(np.mean(scores >= 0.5))
        mean_score = float(np.mean(scores))

        evasion_results.append({
            "dilution_ratio": dilution,
            "detection_rate": round(detection_rate, 4),
            "mean_anomaly_score": round(mean_score, 4),
            "samples_evaded": int(sum(scores < 0.5))
        })

        print(f"Dilution={dilution:.1f} | "
              f"Detection Rate={detection_rate:.2%} | "
              f"Mean Score={mean_score:.4f}")

    with open(RESULTS_DIR / "evasion_curve.json", "w") as f:
        json.dump(evasion_results, f, indent=2)

    return evasion_results


def run_feature_importance_analysis(detector: PromptInjectionDetector):
    """
    Analyse which features most differentiate injections from benign prompts.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Feature Importance Analysis")
    print("="*60)

    extractor = PromptFeatureExtractor()

    benign_features = extractor.transform(BENIGN_PROMPTS)
    injection_features = extractor.transform(INJECTION_PROMPTS)

    benign_mean = benign_features.mean()
    injection_mean = injection_features.mean()

    delta = (injection_mean - benign_mean).abs().sort_values(ascending=False)

    importance_data = []
    for feature in delta.index:
        importance_data.append({
            "feature": feature,
            "benign_mean": round(float(benign_mean[feature]), 4),
            "injection_mean": round(float(injection_mean[feature]), 4),
            "absolute_delta": round(float(delta[feature]), 4)
        })

    print("\nTop 10 Discriminative Features:")
    for row in importance_data[:10]:
        print(f"  {row['feature']:<40} "
              f"benign={row['benign_mean']:.4f}  "
              f"injection={row['injection_mean']:.4f}  "
              f"Δ={row['absolute_delta']:.4f}")

    with open(RESULTS_DIR / "feature_importance.json", "w") as f:
        json.dump(importance_data, f, indent=2)

    return importance_data


def run_category_breakdown(detector: PromptInjectionDetector):
    """
    Break down detection rate by injection category.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Detection by Injection Category")
    print("="*60)

    categories = {
        "direct_override": INJECTION_PROMPTS[0:10],
        "persona_injection": INJECTION_PROMPTS[10:20],
        "delimiter_escape": INJECTION_PROMPTS[20:30],
        "indirect_nested": INJECTION_PROMPTS[30:40],
        "jailbreak_templates": INJECTION_PROMPTS[40:50],
        "data_extraction": INJECTION_PROMPTS[50:60],
    }

    category_results = []
    for category, prompts in categories.items():
        scores = detector.score(prompts)
        detection_rate = float(np.mean(scores >= 0.5))
        mean_score = float(np.mean(scores))

        category_results.append({
            "category": category,
            "n_prompts": len(prompts),
            "detection_rate": round(detection_rate, 4),
            "mean_anomaly_score": round(mean_score, 4)
        })

        print(f"{category:<30} Detection={detection_rate:.0%}  "
              f"Mean Score={mean_score:.4f}")

    with open(RESULTS_DIR / "category_breakdown.json", "w") as f:
        json.dump(category_results, f, indent=2)

    return category_results


def generate_summary(threshold_results, evasion_results, feature_importance, category_results):
    """Compile a single summary JSON for the README and notebook."""

    best = max(threshold_results, key=lambda x: x["f1_injection"])

    summary = {
        "best_threshold": best["threshold"],
        "best_auc_roc": best["auc_roc"],
        "best_f1_injection": best["f1_injection"],
        "best_recall_injection": best["recall_injection"],
        "best_precision_injection": best["precision_injection"],
        "best_accuracy": best["accuracy"],
        "evasion_at_zero_dilution": evasion_results[0]["detection_rate"],
        "evasion_at_50pct_dilution": evasion_results[5]["detection_rate"],
        "evasion_at_90pct_dilution": evasion_results[9]["detection_rate"],
        "top_3_features": [f["feature"] for f in feature_importance[:3]],
        "hardest_category": min(category_results, key=lambda x: x["detection_rate"])["category"],
        "easiest_category": max(category_results, key=lambda x: x["detection_rate"])["category"],
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    detector, scores, labels = run_main_experiment()
    evasion = run_evasion_experiment(detector)
    features = run_feature_importance_analysis(detector)
    categories = run_category_breakdown(detector)
    summary = generate_summary(
        json.load(open(RESULTS_DIR / "threshold_sweep.json")),
        evasion, features, categories
    )
    print("\n[Done] All results saved to /results")
