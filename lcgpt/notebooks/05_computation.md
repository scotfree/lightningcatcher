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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# 05 — Computation

Covers `karpathy.py` lines 19–61: the `Value` class, and the box that notebook 01
asked you to accept without opening.

Nothing above this line is specific to language models. `Value` is a general
reverse-mode automatic differentiator — swap the transformer for anything built out
of the same operations and it would work unchanged. It is the layer that makes
`loss.backward()` on line 224 a single line.

The reason it comes last: every earlier notebook could treat gradients as something
that arrives. This one is where they come from.

Code cells reproduce the source verbatim, dedented where a fragment sits inside a
class. Anything that is *not* from the source is marked as such.

+++

## 5.1 The `Value` node — lines 19–26

Not a number — a node in a graph. Four fields: the scalar itself, a gradient slot
that starts at zero, the nodes this one was computed from, and the derivative of
this node with respect to each of them.

`_local_grads` is the design decision worth noticing. Most implementations of this
store a closure per node — a little function that knows how to propagate. Here the
local derivative is computed during the *forward* pass and stored as a plain
number, so `backward` is arithmetic over data rather than a chain of function
calls.

`__slots__` avoids a per-instance `__dict__`. A single training step on the default
model builds hundreds of thousands of these.

### 5.2 Standard operations — lines 28–34


Addition and multiplication, each recording what it was built from and what its
local derivatives are.

For $c = a + b$ the derivatives are $\partial c/\partial a = 1$ and
$\partial c/\partial b = 1$ — hence `(1, 1)`. Addition passes gradient through
unchanged, which is exactly why the residual connections of notebook 03 keep the
backward path short.

For $c = a \cdot b$ they are $\partial c/\partial a = b$ and
$\partial c/\partial b = a$ — hence `(other.data, self.data)`, each factor's
derivative being the *other* factor.

The `isinstance` line lets a bare float appear on either side of an operator by
wrapping it in a `Value` with no children, which makes it a leaf the graph simply
stops at.

## 5.4 Operators derived from the primitives — lines 40–46

Subtraction, division and negation add no new derivative rules. They are rewrites
into `+` and `*`:

$$
-a = a \cdot (-1)
\qquad
a - b = a + (-b)
\qquad
a / b = a \cdot b^{-1}
$$

So `a / b` builds two nodes — a `__pow__` and a `__mul__` — and the chain rule
handles the quotient rule for free. The `__r*__` variants catch the case where a
plain float is on the left, as in `(1 / n) * sum(losses)` on line 221.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
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

def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

## 5.3 Unary operations — lines 36–39

Four one-liners, each following the same pattern: the forward value, a
single-element children tuple, and the local derivative.

$$
\frac{d}{dx}x^{n} = n x^{n-1}
\qquad
\frac{d}{dx}\log x = \frac{1}{x}
\qquad
\frac{d}{dx}e^{x} = e^{x}
\qquad
\frac{d}{dx}\max(0, x) = [\,x > 0\,]
$$

These four plus `+` and `*` are the entire vocabulary of the model. `log` appears
in the cross-entropy loss, `exp` in the softmax, `relu` in the MLP, and `__pow__`
in `rmsnorm` and in Adam's square root.

ReLU's derivative at exactly zero is a convention — `float(self.data > 0)` picks 0.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

## 5.5 `backward()` — lines 48–61

Two phases.

**Topological sort.** A depth-first walk from the root, appending each node *after*
its children, with a `visited` set so a node reached by several paths is emitted
once. The result is an order in which every node appears after everything it was
built from.

**The reverse sweep.** Seed the root with $\partial \mathcal{L}/\partial \mathcal{L} = 1$,
walk the order backwards, and push each node's gradient down to its children:

$$
\frac{\partial \mathcal{L}}{\partial c} \;\mathrel{+}=\;
\frac{\partial v}{\partial c} \cdot \frac{\partial \mathcal{L}}{\partial v}
$$

That single line is the chain rule. Reversed topological order is what makes it
correct in one pass: when a node is processed, every path from the loss into it has
already been accounted for, so its `grad` is final before it is used.

The `+=` matters. A node used in several places — a weight touched at every
position of a document — receives a contribution from each, and they sum. It is
also why `train` must reset `p.grad = 0` on line 235: nothing here clears anything.

Note that `build_topo` recurses, so graph depth is bounded by Python's recursion
limit. Depth grows with document length, not with corpus size.

```{code-cell} ipython3
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
```

## 5.6 Gradient descent, on its own

Not from the source. The smallest thing that exercises everything above: a scalar
function with a known minimum, descended by hand.

$$
f(x) = (x-3)^2
\qquad
f'(x) = 2(x-3)
$$

We'll use this computational machinery for this simple example, just to see it on its own before getting into the complexities that the GPT model uses it for.

The graph is rebuilt from scratch each step, exactly as the training loop rebuilds
the model's graph for every document — and `x.grad` is cleared each time, for the
reason §5.5 gives.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Not from the source: minimise f(x) = (x - 3)^2, whose minimum is obviously at x = 3.
import sys; sys.path.insert(0, '..')   # karpathy.py lives one level up
from karpathy import Value

x = Value(-2.0)
lr = 0.1

for step in range(30):
    y = (x - 3.0) ** 2   # forward: build a fresh graph
    x.grad = 0           # clear last step's gradient
    y.backward()         # backward: fill in dy/dx
    x.data -= lr * x.grad
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d} | x {x.data:+.5f} | f(x) {y.data:.6f} | df/dx {x.grad:+.5f}")
```

```{code-cell} ipython3

```
