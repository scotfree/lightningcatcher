import math
import random

# Configuration and Computation
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



class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad





# GPT Model and Attention


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def new_model_config(docs, token_type='letter', verbose=True, seed=None, **overrides):
    """Build a tokenizer from `docs` and initialise a fresh set of weights."""
    config = CONFIG_DEFAULTS.copy()
    config.update(overrides)
    config['token_type'] = token_type
    random.seed(seed)
    # Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    config['uchars'] = sorted({t for d in docs for t in doc_to_tokens(d, token_type)}) # unique tokens become ids 0..n-1
    config['head_dim'] = config['n_embd'] // config['n_head']    # derived dimension of each head
    config['vocab_size'] = len(config['uchars']) + 1             # +1 is for BOS
    config['BOS'] = len(config['uchars'])                        # id of the Beginning of Sequence token

    # Initialize the parameters, to store the knowledge of the model
    n_embd, vocab_size, block_size = config['n_embd'], config['vocab_size'], config['block_size']

    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(config['n_layer']):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    config['state_dict'] = state_dict

    if verbose:
        print(f"vocab size: {vocab_size} ({token_type} tokens)")
        print(f"Dimensions: {config['n_layer']} x  {config['n_embd']} x {config['n_embd']} ")
        print(f"Num. Params: {len([p for mat in config['state_dict'].values() for row in mat for p in row])}")
        # Documents longer than the context window are silently cut short by the
        # `n = min(block_size, ...)` in train(). Worth saying out loud on an unfamiliar corpus.
        over = sum(1 for d in docs if len(doc_to_tokens(d, token_type)) + 1 > block_size)
        if over:
            print(f"note: {over} of {len(docs)} documents ({100 * over / len(docs):.0f}%) are longer than "
                  f"block_size={block_size} and will be truncated")
    return config



def gpt(config, token_id, pos_id, keys, values):
    state_dict = config['state_dict']
    n_layer, n_head, head_dim = config['n_layer'], config['n_head'], config['head_dim']

    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Machine Learning - Input
def doc_to_tokens(doc, token_type):
    if token_type == 'letter':
        return list(doc)
    WORD_TRIM = '!"\'(),-.?—'
    cleaned = (w.strip(WORD_TRIM) for w in doc.lower().replace('’', "'").split())
    return [w for w in cleaned if w]

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

    # Repeat in sequence
    for step in range(num_steps):

        # Take single document, tokenize it, surround it with BOS special token on both sides
        # n determines the number of positions for truncation
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(tok) for tok in doc_to_tokens(doc, token_type)] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]

            logits = logit_model(config, token_id, pos_id, keys, values)

            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()

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

        if verbose:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

    if verbose:
        print()
    return config

# Machine Learning - Output
def tokens_to_text(tokens, token_type):
    return ('' if token_type == 'letter' else ' ').join(tokens)

def generate(config, num_samples=20, temperature=0.5, verbose=True, seed=None, logit_model=gpt):
    """Sample from the model. Returns a list of decoded strings.

    `temperature` is in (0, 1] to control the "creativity" of generated text, low to high.
    """
    seed = 42 if seed is None else seed
    random.seed(seed)
    n_layer, block_size, vocab_size = config['n_layer'], config['block_size'], config['vocab_size']
    uchars, BOS, token_type = config['uchars'], config['BOS'], config['token_type']
    if verbose:
        print(f"Generating {num_samples} samples at T={temperature} with seed {seed}")
        
    samples = []
    for sample_idx in range(num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = logit_model(config, token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        text = tokens_to_text(sample, token_type)
        samples.append(text)
        if verbose:
            print(f"sample {sample_idx+1:2d}: {text}")
    return samples




# Model Settings and Computation
# Machine Learning
# GPT Model & Attention

## Define the model architecture: a function mapping tokens and parameters to logits over what comes next
## Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

## The actual language model. Called in training and sampling. 
##  Called on a token and location, and returns logits across all locations.

# The architecture. Overridable via new_config(**overrides), but these are the
# numbers the notebooks describe and the anchor hard-codes.


# Let there be Autograd to recursively apply the chain rule through a computation graph







