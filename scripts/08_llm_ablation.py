"""Zero-shot, RAG and QLoRA ablation of Qwen2.5-1.5B-Instruct (S1 Appendix).

Needs a GPU. Reproduces the appendix table, not any main-text result.
"""
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from cogdml.data import ROOT, SEED, constructs, label, load_survey
from cogdml.metrics import save

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = Path(ROOT) / "prompts"


def intensity(score):
    return ("strong" if score >= 4 else "moderate" if score >= 3
            else "weak" if score >= 2 else "very weak")


def serialize(row, cols):
    means = {k: row[cols[k]].mean() for k in ("ATTD", "SN", "PBC")}
    overall = sum(means.values()) / 3
    template = (PROMPTS / "serialization.txt").read_text()
    return template.format(
        attd=round(means["ATTD"], 2), attd_i=intensity(means["ATTD"]),
        sn=round(means["SN"], 2), sn_i=intensity(means["SN"]),
        pbc=round(means["PBC"], 2), pbc_i=intensity(means["PBC"]),
        overall=round(overall, 2), overall_i=intensity(overall))


def main():
    torch.manual_seed(SEED)
    df = load_survey()
    cols = constructs(df)
    df["text"] = df.apply(lambda r: serialize(r, cols), axis=1)
    df["label"] = label(df)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    save(pd.DataFrame({"split": ["train", "test"], "n": [len(train), len(test)]}),
         "llm_splits.csv")
    print("Serialised prompts ready. Run the paradigm cells on a GPU host; "
          "settings are in prompts/ and docs/llm_settings.md.")


if __name__ == "__main__":
    main()
