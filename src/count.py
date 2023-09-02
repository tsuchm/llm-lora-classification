from pathlib import Path
from tap import Tap
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import src.utils as utils

class Args(Tap):
    output_file: Path = None
    metrics_file: Path = "./outputs/rinna__bilingual-gpt-neox-4b/2023-08-27/06-27-28.787598/test-metrics.json"
    dataset_dir: Path = "./datasets/snli"

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())
        self.category_file: Path = self.dataset_dir / "snli_test_classified.jsonl"

def main(args):
    category: dict[str, str] = {}
    for x in utils.load_jsonl(args.category_file).to_dict(orient="records"):
        category[x['pairID']] = x['clte_class']

    metrics = utils.load_json(args.metrics_file)
    gold_labels: dict[str, list[int]] = defaultdict(list)
    pred_labels: dict[str, list[int]] = defaultdict(list)
    for x in metrics['results']:
        if c := category.get(x['id']):
            gold_labels[c].append(x['gold_label'])
            pred_labels[c].append(x['predicted_label'])
        else:
            print(f"Unknown ID: {x['id']}")

    stat: dict[str, dict[str, float]] = {}
    for c in gold_labels.keys():
        accuracy: float = accuracy_score(gold_labels[c], pred_labels[c])
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels[c],
            pred_labels[c],
            average="macro",
            zero_division=0,
            labels=args.labels,
        )
        stat[c] = {"number": len(gold_labels[c]),
                   "accuracy": accuracy,
                   "precision": precision,
                   "recall": recall,
                   "f1": f1}

    if args.output_file:
        utils.save_json(stat, args.output_file)
    else:
        print(stat)

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
