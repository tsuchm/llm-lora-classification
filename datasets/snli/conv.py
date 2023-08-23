#!/usr/bin/python3
# -*- coding: utf-8; -*-

import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_args():
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument('inputs', metavar='FILE', type=str, nargs='+',
                   help='a JSON formatted file')
    p.add_argument('--output', type=ap.FileType('w'), default=sys.stdout)
    p.add_argument('--labeldic', type=str, help='en')
    p.add_argument('--language', type=str, help='en')
    return p.parse_args()

def convirt(input, output, labeldic, default_language='en'):
    for line in input:
        x = json.loads(line)
        if x['gold_label'] in labeldic:
            new = {'title': x['sentence1'],
                   'body': x['sentence2'],
                   'id': x['pairID'],
                   'label': labeldic[x['gold_label']]}
            print(json.dumps(new, ensure_ascii=False), file=output)

if __name__ == '__main__':
    args = parse_args()
    with open(args.labeldic, 'r', encoding='utf-8', errors='ignore') as fp:
        labeldic = json.load(fp)
    for file in args.inputs:
        with open(file, 'r', encoding='utf-8', errors='ignore') as fp:
            convirt(fp, args.output, labeldic, default_language=args.language)
