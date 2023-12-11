from pathlib import Path
from tap import Tap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
from datasets import load_dataset

import src.utils as utils

class Args(Tap):
    model_name: str = "google/flan-t5-xl"
    dataset_dir: Path = "./datasets/snli"
    output_file: Path = None
    device: int = 0 if torch.cuda.is_available() else -1
    batch_size: int = 16
    ignore_premise: bool = False

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(self.dataset_dir / "label2id.json")
        self.labels: list[int] = list(self.label2id.values())
        self.test_file: Path = self.dataset_dir / "test.jsonl"

def main(args):
    dataset = load_dataset('json', data_files={'test': str(args.test_file)}, split='test')
    def dsconv(x):
        if args.ignore_premise:
            x['text'] = """question: Classify the logical relationship between the two sentences given as the context into three categories, such as entailment, neutral and contradiction.
context: """ + x['body']
        else:
            x['text'] = """question: Classify the logical relationship between the two sentences given as the context into three categories, such as entailment, neutral and contradiction.
context: """ + x['title'] + "\n" + x['body']
        return x
    dataset = dataset.map(dsconv)

    pipe = pipeline(model=args.model_name,
                    task='text2text-generation',
                    device=args.device)

    gold_labels = []
    pred_labels = []
    results = []
    for e,r in zip(dataset,
                   pipe(KeyDataset(dataset, 'text'),
                        batch_size=args.batch_size)):
        #print(e['text'])
        #print(r[0]['generated_text'])
        p = args.label2id.get(r[0]['generated_text'], -1)
        if p >= 0:
            pred_labels.append(p)
            gold_labels.append(e['label'])
            results.append({'id': e['id'],
                            'gold_label': e['label'],
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
