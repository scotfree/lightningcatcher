#!/usr/bin/env python3
"""A command-line and importable wrapper around karpathy_gpt.py.

karpathy_gpt.py is this project's fixed reference artifact and is never edited:
the notebooks quote it by line number. This file is that same algorithm, made
importable and given a few conveniences.

As a script it behaves exactly as before:

    --num-steps N    how many training steps to run   (alias: --training-runs)
    --num-docs N     use only the first N documents
    --corpus-file P  train on P instead of input.txt, one document per line
    --token-type T   'letter' (default, what the anchor does) or 'word'
    --model PATH     load a saved model and sample from it; if PATH does not
                     exist, train and save there instead

    python karpathy_gpt_cli.py --num-steps 200 --num-docs 5000
    python karpathy_gpt_cli.py --corpus-file data/beatles_first3.txt --token-type word
    python karpathy_gpt_cli.py --model model.json

As a library, a "config" is a plain dict carrying the hyperparameters, the
tokenizer and the weights together -- they are meaningless apart, since token ids
only mean anything relative to the `uchars` they were trained against:

    import karpathy_gpt_cli as lc

    config = lc.get_model('beatles_word1.json')     # loads, or trains and saves
    lc.generate(config, num_samples=5)

    docs   = lc.load_corpus('data/beatles_first3.txt', num_docs=500)
    config = lc.new_config(docs, token_type='word')
    lc.train(config, docs, num_steps=200)
    lc.save_model('mine.json', config)

Still dependency-free: argparse and json are standard library.
"""

import os
import json
import math
import random
import argparse

import karpathy



# random.seed(42) # Let there be order among chaos. Note: this runs at import time.


def derive(config):
    """Fill in the values implied by the rest of the config.

    A config is a plain dict, so these cannot recompute themselves. Every function
    that creates or alters `uchars`, `n_embd` or `n_head` calls this, which is what
    keeps them from drifting out of sync with the weights.
    """
    
    return config



def load_docs_textfile(path, num_docs=None, shuffle=True, verbose=True):
    """Read a corpus of one document per line.

    The default corpus is fetched on first use; any other path must already exist,
    so a typo fails loudly instead of silently downloading names over the top of it.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no such file")

    docs = [line.strip() for line in open(path) if line.strip()]
    if shuffle:
        random.shuffle(docs)
    if num_docs is not None:
        docs = docs[:num_docs]
    if verbose:
        print(f"corpus: {path}")
        print(f"num docs: {len(docs)}")
    return docs



SAVED_KEYS = ('n_layer', 'n_embd', 'block_size', 'n_head', 'vocab_size', 'token_type', 'BOS', 'head_dim', 'vocab_size')

def save_model(path, config, verbose=True):
    blob = {
        'format': 'lcgpt-1',
        'config': {k: config[k] for k in SAVED_KEYS},
        'uchars': config['uchars'],
        'state_dict': {name: [[p.data for p in row] for row in mat]
                       for name, mat in config['state_dict'].items()},
    }
    with open(path, 'w') as f:
        json.dump(blob, f)
    if verbose:
        print(f"saved model to {path}")
    return path

def load_model(path, verbose=True):
    """Read a model file back into a config dict."""
    with open(path) as f:
        blob = json.load(f)
    if blob.get('format') != 'lcgpt-1':
        raise SystemExit(f"{path}: not an lcgpt-1 model file")

    config = karpathy.CONFIG_DEFAULTS
    config.update(blob['config'])
    config.setdefault('token_type', 'letter') # files written before the flag existed
    config['uchars'] = blob['uchars']
    config['state_dict'] = {name: [[karpathy.Value(x) for x in row] for row in mat]
                            for name, mat in blob['state_dict'].items()}
    derive(config) # recompute rather than trust the file, so the two cannot disagree
    if verbose:
        print(f"loaded model from {path}")
    return config

    

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-steps', '--training-runs', type=int, default=1000, dest='num_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--seed',  type=int, default=43, dest='seed',
                            help='RNG seed')
    parser.add_argument('--num-docs', type=int, default=None, dest='num_docs',
                        help='use only the first N documents after shuffling (default: all)')
    parser.add_argument('--temperature', type=float, default=0.6, dest='temperature',
                            help='Temperature')
    parser.add_argument('--num-samples', type=int, default=10, dest='num_samples',
                            help='generate N emissions fro GPT')
    parser.add_argument('--corpus-file', type=str, default=None, dest='corpus_file',
                        help=f'training corpus, one document per line')
    parser.add_argument('--token-type', choices=('letter', 'word'), default='letter', dest='token_type',
                        help='how documents are split into tokens (default: letter)')
    parser.add_argument('--model', type=str, default='model.json', dest='model_path',
                        help=f'model file; loaded if it exists, otherwise written after training ')
    return parser

def main(argv=None, verbose=True):
    args = build_parser().parse_args(argv)
    if args.model_path and not args.corpus_file:
        model =  load_model(args.model_path, verbose=verbose)
    else:
        if verbose:
            print(f"training a new model on {args.corpus_file}")
        docs = load_docs_textfile(args.corpus_file, num_docs=args.num_docs, shuffle=True, verbose=verbose)
        config = karpathy.new_model_config(docs, token_type=args.token_type)
        #model = get_model(config, model_path=args.model_path, corpus_file=args.corpus_file,
        #                       token_type=args.token_type, num_steps=args.num_steps, num_docs=args.num_docs)
        model = karpathy.train(config, docs, num_steps=args.num_steps, verbose=verbose)
        if args.model_path:
            save_model(args.model_path, model, verbose=verbose)
        # return model
    
    print("--- inference (new, hallucinated documents) ---")
    karpathy.generate(model, num_samples=args.num_samples,temperature=args.temperature, seed=args.seed)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
