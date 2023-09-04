from pathlib import Path
from tap import Tap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
from datasets import load_dataset

import src.utils as utils

class Args(Tap):
    model_name: str = "facebook/bart-large-mnli"
    dataset_dir: Path = "./datasets/snli"
    output_file: Path = None
    device: int = 0 if torch.cuda.is_available() else -1
    batch_size: int = 16

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())
        self.test_file: Path = self.dataset_dir / "test.jsonl"

def main(args):
    dataset = load_dataset('json', data_files={'test': str(args.test_file)}, split='test')
    def dsconv(x):
        x['text'] = x['title'] + "\n" + x['body']
        return x
    dataset = dataset.map(dsconv)

    classifier = pipeline(model=args.model_name, device=args.device)

    gold_labels = []
    pred_labels = []
    results = []
    for example,result in zip(dataset,
                              classifier(KeyDataset(dataset, 'text'),
                                         batch_size=args.batch_size,
                                         candidate_labels=list(args.label2id.keys()))):
        p = args.label2id[result['labels'][0]]
        pred_labels.append(p)
        gold_labels.append(example['label'])
        results.append({'id': example['id'],
                        'gold_label': example['label'],
                        'predicted_label': p})

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
