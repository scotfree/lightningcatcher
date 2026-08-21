---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: LightningcatcherGPT
  language: python
  name: lcgpt
---

# 02 — Input

Covers `karpathy.py` lines 148–242, the *Machine Learning - Input* section, plus
`CONFIG_DEFAULTS` at lines 5–17: turning a pile of documents into tokens, and
turning tokens into weights.

Notebook 01 sampled from a model that was handed to it. This one builds the two
things that model needed — a vocabulary, and a training loop that moves parameters
downhill. The loop is written against `logit_model`, exactly as `generate` was, so
the last section trains something you can print in full.

Code cells reproduce the source verbatim, dedented where a fragment sits inside a
function. Anything that is *not* from the source is marked as such.

+++

## 2.1 Splitting a document into tokens — lines 149–154

A *token* is whatever unit the model treats as indivisible. Two choices here:

- **`letter`** — one character per token. `list(doc)` and nothing more.
- **`word`** — whitespace-separated, lowercased, with the curly apostrophe folded
  onto the straight one and punctuation trimmed off each end, so that `"Yeah,"`,
  `"yeah"` and `"yeah!"` collapse to a single token rather than three.

The choice changes only how many distinct tokens exist. Nothing downstream can
tell the difference.

```{code-cell} ipython3
def doc_to_tokens(doc, token_type):
    if token_type == 'letter':
        return list(doc)
    WORD_TRIM = '!"\'(),-.?—'
    cleaned = (w.strip(WORD_TRIM) for w in doc.lower().replace('’', "'").split())
    return [w for w in cleaned if w]
```

## 2.2 The default configuration — lines 5–17

Every dimension of the model, in one dict. `n_embd` is the width of a token's
vector, `n_layer` the number of stacked blocks, `block_size` the furthest back
attention can look, and `n_head` how many independent attention channels split
that width. The remaining four are the optimiser's.

`block_size=16` is not arbitrary: the longest name in `data/names.txt` is 15
characters, plus one `BOS`.

```{code-cell} ipython3
CONFIG_DEFAULTS = {
    'n_layer': 1,     # depth of the transformer neural network (number of layers)
    'n_embd': 16,     # width of the network (embedding dimension)
    'block_size': 16, # maximum context length of the attention window
    'n_head': 4,      # number of attention heads
    'token_type': 'letter',
    'learning_rate': 0.01, 
    'beta1': 0.85, 
    'beta2': 0.99, 
    'eps_adam': 1e-8
}


```

## 2.3 Building the vocabulary — lines 156–165

Every distinct token in the corpus, sorted, becomes an integer id. Sorting matters
— it makes the mapping reproducible across runs, which is why a saved model can
ship its `uchars` and get the same ids back.

`BOS` takes the one id past the end. It is a real token to the model, and it does
double duty: it starts every sequence and it ends every sequence, so "the document
is finished" is something the model predicts rather than something imposed on it.

```{code-cell} ipython3
def new_tokenizer(docs, token_type='letter', **overrides):
    """Build a config and its tokenizer from `docs`."""
    config = CONFIG_DEFAULTS.copy()
    config.update(overrides)
    config['token_type'] = token_type
    # Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    config['uchars'] = sorted({t for d in docs for t in doc_to_tokens(d, token_type)}) # unique tokens become ids 0..n-1
    config['vocab_size'] = len(config['uchars']) + 1             # +1 is for BOS
    config['BOS'] = len(config['uchars'])                        # id of the Beginning of Sequence token
    return config
```

## 2.4 Tokenizer plus weights — lines 167–183

A convenience wrapper: build the vocabulary, then hang a fresh set of weights off
it. `init_params` is notebook 03's subject. Nothing in these notebooks calls this
function — it exists for the command line — but it is the one place that shows the
two halves belong to one object.

The truncation warning matters on an unfamiliar corpus. Documents longer than
`block_size` are silently cut short in the loop below, and on a corpus of song
lyrics rather than names that is most of them.

```{code-cell} ipython3
def new_model_config(docs, token_type='letter', verbose=True, seed=None, **overrides):
    """Build a tokenizer from `docs` and initialise a fresh set of weights."""
    config = new_tokenizer(docs, token_type, **overrides)
    init_params(config, seed)
    if verbose:
        block_size = config['block_size']
        print(f"vocab size: {config['vocab_size']} ({token_type} tokens)")
        print(f"Dimensions: {config['n_layer']} x  {config['n_embd']} x {config['n_embd']} ")
        print(f"Num. Params: {len([p for mat in config['state_dict'].values() for row in mat for p in row])}")
        # Documents longer than the context window are silently cut short by the
        # `n = min(block_size, ...)` in train(). Worth saying out loud on an unfamiliar corpus.
        over = sum(1 for d in docs if len(doc_to_tokens(d, token_type)) + 1 > block_size)
        if over:
            print(f"note: {over} of {len(docs)} documents ({100 * over / len(docs):.0f}%) are longer than "
                  f"block_size={block_size} and will be truncated")
    return config

```

## 2.5 What training needs — lines 185–198

Hyperparameters out of the config, then the flat list of every parameter in the
model — `params` is what the optimiser walks, and its order is fixed by dict
insertion order so the two Adam buffers can be plain parallel lists.

`m` and `v` are those buffers, one slot per parameter: a running average of the
gradient, and a running average of its square. They start at zero and persist
across steps, which is the whole difference between Adam and plain gradient
descent.

```{code-cell} ipython3
def train(config, docs, num_steps=1000, verbose=True, logit_model=gpt,  **hyper):
    """Train `config` in place on `docs`, one document per step. Returns the config."""
    #settings = dict(TRAIN_DEFAULTS)
    config.update(hyper)
    learning_rate = config['learning_rate']
    beta1, beta2, eps_adam = config['beta1'], config['beta2'], config['eps_adam']

    uchars, BOS = config['uchars'], config['BOS']
    block_size, n_layer, token_type = config['block_size'], config['n_layer'], config['token_type']

    # Let there be Adam, the blessed optimizer and its buffers
    params = [p for mat in config['state_dict'].values() for row in mat for p in row]
    m = [0.0] * len(params) # first moment buffer
    v = [0.0] * len(params) # second moment buffer
```

## 2.6 One document at a time — lines 200–207

No batching. One step, one document, cycling with `%` so the corpus repeats.

The token sequence is wrapped in `BOS` at both ends, then `n` caps the number of
positions at `block_size`. Note what `min` does on a long document: it truncates
silently. The tail is not trained on.

```{code-cell} ipython3
# Repeat in sequence
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    # n determines the number of positions for truncation
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(tok) for tok in doc_to_tokens(doc, token_type)] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

## 2.7 The forward pass — lines 209–216

A fresh KV cache per document, then one call to the model per position. At
position `pos_id` the model sees `tokens[pos_id]` and is asked to predict
`tokens[pos_id + 1]`.

This is what makes the corpus self-supervising: no labels exist, because every
token is the label for the one before it. A document of length $n$ yields $n$
training signals, not one.

`logit_model` is the same seam `generate` used — the loop never names the
transformer.

```{code-cell} ipython3
# Forward the token sequence through the model, building up the computation graph all the way to the loss

keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
losses = []
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]

    logits = logit_model(config, token_id, pos_id, keys, values)
```

## 2.8 Cross-entropy — lines 218–221

The logits become probabilities, and the loss is the negative log of the
probability the model assigned to the token that actually came next:

$$
\mathcal{L} \;=\; -\frac{1}{n} \sum_{t=0}^{n-1} \log p_{\theta}\!\left(x_{t+1} \mid x_{\le t}\right)
$$

Only the target's probability appears. Everything else enters through the
denominator of the softmax, which is what stops the model from simply predicting
every token at once.

A useful reference point: a model that has learned nothing spreads its mass evenly
and scores $\log V$ — about 3.30 for the 27-token names vocabulary. That is the
number the loss starts at in the demo below.

```{code-cell} ipython3
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
```

## 2.9 Backward — lines 223–224

One line, and notebook 05 is entirely about what it does. `loss` is the root of a
graph containing every operation performed above; `backward()` walks that graph in
reverse and leaves `p.grad` on every parameter.

```{code-cell} ipython3
# Backward the loss, calculating the gradients with respect to all model parameters
loss.backward()
```

## 2.10 The Adam update — lines 226–235

Plain gradient descent would be `p.data -= lr * p.grad`. Adam keeps two running
averages instead, and divides the first by the square root of the second:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}
\qquad
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The effect is a per-parameter step size: a parameter with consistently small
gradients still moves, because the division rescales it. The $1-\beta^t$ terms
correct for the buffers starting at zero, and matter only in the first few steps.

`lr_t` decays linearly to zero across the run, and `p.grad = 0` clears the
gradient for the next step — the graph is rebuilt from scratch each document, but
the gradients accumulate with `+=` and would otherwise carry over.

```{code-cell} ipython3
# Adam optimizer update: update the model parameters based on the corresponding gradients
# Note this is pure ML, no GPT/etc?
lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

## 2.11 Watching it train — lines 237–242

The carriage return keeps the loss on one line. `train` mutates `config` in place
and returns it.

```{code-cell} ipython3
    if verbose:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

if verbose:
    print()
return config
```

## 2.12 A model made of parameters

Not from the source. Notebook 01 built a model out of counting. This one has the
same shape — one row of logits per current token — but the numbers are `Value`
objects that training is allowed to move:

```python
config['state_dict'] = {'bigram': [[Value(0.0) for _ in range(V)] for _ in range(V)]}
```

That is the whole model: a $V \times V$ table, no attention, no embeddings, no
layers. It satisfies `logit_model`, so `train` accepts it unmodified — the param
flattening on line 196 walks a `state_dict` of any shape.

Starting every entry at `0.0` means the model begins perfectly uniform, so the
loss starts at $\log 27 \approx 3.30$.

There is a fact worth checking at the end. Cross-entropy is minimised, for a model
this shape, exactly when its distribution equals the corpus's own conditional
frequencies. So the table it *learns* by gradient descent should converge to the
table notebook 01 built by *counting*:

$$
\operatorname{softmax}(\theta_a)_b \;\longrightarrow\; \frac{C(a, b)}{\sum_j C(a, j)}
$$

Nothing in the training loop knows about counting. The agreement is what the
gradient is for.

```{code-cell} ipython3
# Not from the source: the smallest model with parameters in it.
import sys; sys.path.insert(0, '..')   # karpathy.py and lcgpt.py live one level up

from collections import Counter
import karpathy
from karpathy import Value, new_tokenizer, train, generate, softmax
import lcgpt

docs = lcgpt.load_docs_textfile('../data/names.txt', num_docs=500, seed=42, verbose=False)

# a tokenizer with no model attached to it at all
config = new_tokenizer(docs, n_layer=0, learning_rate=0.1)
uchars, BOS, V = config['uchars'], config['BOS'], config['vocab_size']
print("new_tokenizer gives:", sorted(config))
print("...and no state_dict:", 'state_dict' not in config, "\n")

# the entire model
config['state_dict'] = {'bigram': [[Value(0.0) for _ in range(V)] for _ in range(V)]}

def bigram(config, token_id, pos_id, keys, values):
    return config['state_dict']['bigram'][token_id]

train(config, docs, num_steps=4000, logit_model=bigram, verbose=False)

# the same table, counted rather than learned
counts = Counter()
for d in docs:
    seq = [BOS] + [uchars.index(c) for c in karpathy.doc_to_tokens(d, 'letter')] + [BOS]
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] += 1

worst = 0.0
for a in range(V):
    total = sum(counts[(a, j)] for j in range(V))
    if total < 50:
        continue
    learned = [p.data for p in softmax(config['state_dict']['bigram'][a])]
    worst = max(worst, max(abs(counts[(a, j)] / total - learned[j]) for j in range(V)))
print(f"learned vs counted, worst probability gap: {worst:.4f}\n")

generate(config, num_samples=8, temperature=0.8, seed=3, logit_model=bigram)
```
