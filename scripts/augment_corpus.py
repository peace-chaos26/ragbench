import json
import os
import random
from datetime import datetime

BASE_DOCS = [
    "Helm templates are rendered using values from values.yaml and the Go template language.",
    "Helm uses Go templating; values are injected via .Values and can be overridden via --set or values files.",
    "Terraform state stores information about managed infrastructure and can be local or remote.",
    "A Kubernetes Deployment manages a ReplicaSet to keep a set of Pods running.",
]

TOPIC_NEIGHBORS = [
    "Kubernetes Services provide stable networking endpoints for Pods.",
    "ConfigMaps store non-secret configuration data in key-value pairs.",
    "Secrets store sensitive data like tokens and passwords.",
    "kubectl applies manifests to create or update cluster resources.",
    "Helm charts package Kubernetes resources for easy installation.",
    "Terraform providers enable managing resources across cloud platforms.",
    "Terraform plan shows proposed infrastructure changes before apply.",
    "A ReplicaSet maintains a stable set of replica Pods.",
    "Deployments support rolling updates and rollbacks.",
    "Helm values can be layered across multiple values files.",
]

# Wrong-but-plausible statements designed to create adversarial semantic matches
ADVERSARIAL_FALSES = [
    "Terraform uses values.yaml to configure its modules by default.",
    "Helm stores state in terraform.tfstate for tracking releases.",
    "Kubernetes Deployments manage Services directly to keep Pods running.",
    "Terraform templates are rendered using Go templating and values.yaml.",
    "Helm is written in Rust and uses Cargo for building charts.",
    "Terraform state is optional and not needed for planning changes.",
    "Helm uses HCL files and providers, similar to Terraform.",
    "Kubernetes uses values.yaml as the default manifest format.",
]

# Simple rule-based “paraphrases” (not perfect, but increases variety)
REWRITES = [
    ("uses", "leverages"),
    ("stores", "keeps"),
    ("manages", "controls"),
    ("is used for", "helps with"),
    ("can be", "may be"),
    ("overridden", "changed"),
    ("configuration", "settings"),
    ("injected", "passed"),
    ("templates", "manifests"),
    ("running", "active"),
]

def rewrite_sentence(s: str, rng: random.Random, n_swaps: int = 2) -> str:
    out = s
    swaps = rng.sample(REWRITES, k=min(n_swaps, len(REWRITES)))
    for a, b in swaps:
        out = out.replace(a, b)
    # small noise: optional prefix/suffix
    if rng.random() < 0.3:
        out = "Note: " + out
    if rng.random() < 0.3:
        out = out + " (high level)"
    return out

def make_distractors(n: int, seed: int = 42):
    rng = random.Random(seed)
    docs = []

    # Mix of neighbor docs, adversarial falses, and rewritten variants
    while len(docs) < n:
        r = rng.random()

        if r < 0.45:
            # topical neighbor (semantically close, not answer-supporting)
            base = rng.choice(TOPIC_NEIGHBORS)
            docs.append(rewrite_sentence(base, rng, n_swaps=1))

        elif r < 0.70:
            # adversarial false (highly confusing)
            base = rng.choice(ADVERSARIAL_FALSES)
            docs.append(rewrite_sentence(base, rng, n_swaps=1))

        else:
            # rewritten variant of a true doc (near-duplicate noise)
            base = rng.choice(BASE_DOCS)
            docs.append(rewrite_sentence(base, rng, n_swaps=3))

    return docs

def main():
    os.makedirs("data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    distractors = make_distractors(1000, seed=123)
    all_docs = BASE_DOCS + distractors

    out_path = f"data/corpus_augmented_{ts}.jsonl"
    with open(out_path, "w") as f:
        for i, text in enumerate(all_docs):
            row = {
                "id": f"doc_{i}",
                "text": text,
                "source": "base" if i < len(BASE_DOCS) else "synthetic",
            }
            f.write(json.dumps(row) + "\n")

    print("Wrote:", out_path)
    print("Docs:", len(all_docs), "(base:", len(BASE_DOCS), "synthetic:", len(distractors), ")")

if __name__ == "__main__":
    main()