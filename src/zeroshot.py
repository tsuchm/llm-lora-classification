from pathlib import Path
from tap import Tap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import pipeline

import src.utils as utils

class Args(Tap):
    model_name: str = "facebook/bart-large-mnli"
    output_file: Path = None
    dataset_dir: Path = "./datasets/snli"

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())
        self.test_file: Path = self.dataset_dir / "test.jsonl"

def main(args):
    classifier = pipeline(model=args.model_name,
                          device=0 if torch.cuda.is_available() else -1)

    gold_labels = []
    pred_labels = []
    results = []
    for x in utils.load_jsonl(args.test_file).to_dict(orient="records"):
        r = classifier(x['title'] + "\n" + x['body'],
                       candidate_labels=list(args.label2id.keys()))
        p = args.label2id[r['labels'][0]]
        pred_labels.append(p)
        gold_labels.append(x['label'])
        results.append({'id': x['id'], 'gold_label': x['label'], 'predicted_label': p})

    accuracy: float = accuracy_score(gold_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels,
        pred_labels,
        average="macro",
        zero_division=0,
        labels=args.labels,
    )

    stat: dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "results": results
    }
    if args.output_file:
        utils.save_json(stat, args.output_file)
    else:
        print(stat)

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
