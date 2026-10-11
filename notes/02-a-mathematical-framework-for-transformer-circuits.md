# 02 — A Mathematical Framework for Transformer Circuits

_Elhage et al., 2021 (transformer-circuits.pub). Read Sep 25 – Oct 5, 2026. I read up to **Term Importance Analysis** and stopped; the sections after it didn't look like they'd add anything for this project._

A lot of these notes are me building up the basics from the paper, not just the paper's results. I've kept my wrong turns in where they taught me something. This is exactly the part that I really enjoy: get something, start from the most basic building blocks and build the understanding up again. There is always something new to learn, and you often find you'd understood things differently before (not necessarily misunderstood, but the knowledge could become compressed and more intuitive).

---

## 1. The simplifications the paper makes

- **No biases.** A bias can be folded into the weights by adding a dimension that's always 1. `y = Wx + b` becomes `y = W'x'`, where `x' = [x, 1]` and `b` is the last column of `W'`. So dropping biases loses nothing.
- **Attention-only models** (no MLPs) for the small models they actually reverse-engineer. My model is the same: 2 layers, attention only.
- **No LayerNorm.** I'm skipping it too.

## 2. The residual stream is a communication channel

The residual stream is the one vector per position that runs from the embedding to the unembedding. Every component interacts with it in exactly two ways:

- **reads** from it by multiplying by a matrix (W_Q, W_K, W_V, an MLP's W_in, W_U)
- **writes** to it by multiplying its output by a matrix and **adding** it in (W_O, W_out, W_E)

Nothing ever overwrites the stream. It only gets added to, and its size never changes from start to finish.

**Subspaces as lanes.** This took me a while. The way I understand it now: each component writes into some set of directions and each reader is only sensitive to some set of directions. If a writer and a reader agree on a direction, information goes through. If a direction is perpendicular to what a reader reads, that reader can't see it at all. So two components can write to the same stream without interfering, as long as they use perpendicular directions. Small 2D check that convinced me:

```
u = [0.6, 0.8], v = [0.8, -0.6]   (perpendicular)
layer 1 writes 5 along u  -> [3, 4]
layer 2 writes 2 along v  -> [1.6, -1.2]
stream = [4.6, 2.8]
read along u: 0.6*4.6 + 0.8*2.8 = 5.0   <- layer 1's message
read along v: 0.8*4.6 - 0.6*2.8 = 2.0   <- layer 2's message
```

Neither message sits in "its own" coordinate, but both come back exactly.

**Embeddings use a small subspace.** The paper says the embedding and unembedding only use a small part of the stream, which leaves the rest free as scratch space for the layers. My first instinct was that every token should get its own plane because tokens mean different things. That's wrong: all token embeddings are combinations of the *same* small set of directions (different amounts per token). The "plane" isn't something the model stores; it's just a description of where the trained embedding vectors ended up. Nothing gets zeroed. The free directions are simply ones the embedding doesn't point along.

My project is exactly this: the previous-token head writes "the token before me was X" into free directions, and the induction head later reads it from there.

**What happens to junk?** If a head writes noise somewhere, nothing erases it (the stream only gets added to). Two things deal with it: later layers learn through training to never read it, and heads with nothing to do learn to "park" their attention on a harmless position (usually the first token) whose value is ~0, so they write almost nothing. The softmax can't output "attend to nothing", which is why the parking exists.

How do a writer and a reader "agree" on a direction? Nobody picks it. Both start random and backprop nudges them into agreement over training.

## 3. No privileged basis

The paper says the residual stream has no privileged basis: you can rotate it, by rotating every matrix that touches it, without changing the model's behaviour.

- **Basis** = the axes you write coordinates in.
- **Privileged** = something in the architecture treats coordinates individually, so the axes mean something.
- **Rotating**: pick an orthogonal R. Every write matrix becomes `R · W_write`, every read matrix becomes `W_read · Rᵀ`. Any read-after-write path computes `W_read Rᵀ R W_write = W_read W_write`. Same numbers downstream, same outputs.

The rotation is just a proof device: it shows the stream's axes can't mean anything. Why that matters: if "coordinate 37 = feature X" is true in one model and false in an equally good rotated copy that computes the same thing, then it was never a fact about the model. The meaningful objects are the things that don't change under rotation, like `W_Qᵀ W_K` and `W_O W_V`, which is why the paper builds everything on them.

**Where I got confused: ReLU.** I thought the MLP's ReLU would break this, since ReLU is per-coordinate. It doesn't, because the ReLU never acts on the residual stream. It acts on the MLP's hidden neurons, between the read (W_in) and the write (W_out). W_in "un-rotates" before the hidden neurons, so they see the same numbers as before. So: hidden MLP neurons **are** a privileged basis; the residual stream isn't. The ReLU affects *what gets written* to the stream, but never acts *on* the stream's coordinates.

Side link worth remembering: quantization methods like QuaRot and SpinQuant use exactly this invariance. They rotate the stream to spread outlier dimensions out before quantizing. Same model, friendlier numbers.

## 4. Attention heads: QK and OV

The "conceptual takeaways" bullet I originally couldn't parse says a head is really two independent computations:

- **QK circuit (`W_Qᵀ W_K`)**: how much does position i attend to position j? Uses properties of *both* tokens.
- **OV circuit (`W_O W_V`)**: if i attends to j, what does j contribute to i, and where is it written? Uses properties of *j only*.

My first answer was that OV decides "whether we want this information". Wrong: that decision is entirely QK's. OV is a fixed rule, "whatever gets attended to, transform it this way and write it here", scaled by the attention weight.

Query / key / value, in the words I'd use now:
- **query**: computed at the position doing the looking; describes what it's looking for
- **key**: computed at each position being looked at; a label used only for matching against queries
- **value**: what actually gets copied over if that position gets attention. It can be completely different from the key (catalog entry vs the book's contents).

The paper's point is that q, k and v are just intermediate results. A head is fully described by two matrices.

**Heads vs CNN kernels.** I compared a head to a CNN kernel. The attention pattern is a bit like a kernel's weights, except recomputed from content every time and able to look anywhere earlier. But a kernel *detects*; a head mostly *moves* features between positions. Detection is the MLPs' job. My attention-only model can only move information around.

## 5. The causal mask, and my "why not revise?" idea

Attention only writes to the position it's computed at, and only reads earlier positions. I asked why we don't let later tokens revise earlier ones. Turns out this exists: BERT-style encoders drop the causal mask (trained by predicting blanked-out words instead of next tokens), and diffusion-style language models generate all positions at once and revise over steps. Causal models win for generation because (1) every position is a valid training prediction in one pass, and (2) earlier positions never change at inference, so their computation can be cached.

## 6. Tensor notation and the one-layer model

- `Id ⊗ W` = transform each position's vector by W on its own (`X Wᵀ`). No position sees another.
- `A ⊗ Id` = mix positions, weighted by A (`A X`). Vectors get mixed but not transformed.
- I guessed the second one was element-wise. It isn't: it keeps feature dimensions separate, but it sums across positions, so it's still a matrix multiply.
- `A ⊗ W` = `A X Wᵀ`, and the order doesn't matter.
- Mixed-product rule: `(A₂ ⊗ W₂)(A₁ ⊗ W₁) = (A₂A₁) ⊗ (W₂W₁)`. Position mixing stacks with position mixing; vector transforms stack with vector transforms.

**Why the product is written "in reverse".** I thought it was because we reverse-engineer from the logits. It's just function composition: the input sits on the right, so the rightmost factor acts first. Read right to left, it's the forward pass in order.

**Freezing attention.** The attention pattern A is the only non-linear piece (it's a softmax of token-dependent scores). If you run the model once, record every A, and treat them as constants, everything else is matrix multiplies and additions, so the logits become linear in the tokens. My version: tokens → QK circuit → **[freeze]** → OV circuit → logits.

**One-layer models learn skip-trigrams** (`[b] … [a] → [b]`): a lookup table of "if this earlier token, and this current token, boost this next token". The "R" / "alph" example confused me because it looks like the head predicts forward. It doesn't: the head at "R" attends back to an earlier " Ralph" token (one token holding the whole word), and its OV maps " Ralph" → "alph". It only works when one earlier token already holds the answer.

## 7. Two-layer models

### Path expansion of the logits

Write the model as a chain, `Id ⊗ Id` being the residual connection:

```
x0 = (Id ⊗ W_E) t
x1 = (Id ⊗ Id + Σ_{h∈H1} A^h ⊗ W_OV^h) x0
x2 = (Id ⊗ Id + Σ_{h∈H2} A^h ⊗ W_OV^h) x1
T  = (Id ⊗ W_U) x2
```

Multiplying out `(1 + b)(1 + a) = 1 + a + b + ba` gives three kinds of path:

1. **Direct path** `Id ⊗ W_U W_E`: no movement between positions, so bigrams only.
2. **Individual heads** `A^h ⊗ W_U W_OV^h W_E`: one head, one hop.
3. **Virtual heads** `(A^h2 A^h1) ⊗ (W_U W_OV^h2 W_OV^h1 W_E)`: a layer-1 head then a layer-2 head.

"Virtual" because term 3 has the exact shape of a single head (position mixing ⊗ vector transform), but the head exists in no weight matrix. `A^h2 A^h1` is the right pattern because moving k → j and then j → i, summed over every j, is literally matrix multiplication. A virtual head only exists if h2's value reads the subspace h1 writes to. That overlap is **V-composition**. Virtual heads barely matter in this small model; they're expected to matter more in big ones (e.g. "look two tokens back" from two previous-token hops).

**The catch:** this equation freezes every A, but a layer-2 head computes its A from `x1`, which contains layer-1 output. So layer 1 can change *where layer 2 looks*, and that's hidden inside the symbol `A^h2`.

### Three kinds of composition

A layer-2 head reads the stream through three matrices. Composition = one of them reads something a layer-1 head wrote:

| Type | Reads layer 1 via | Layer 1 influences |
|---|---|---|
| Q-composition | W_Q | what the destination looks for |
| K-composition | W_K | what each source advertises about itself |
| V-composition | W_V | what gets moved (→ virtual heads) |

Q and K change the attention pattern itself, which is why they can't be folded into "one more head".

### Query side and key side

I assumed this was about tensor notation. It isn't. It's about where the two inputs to the dot product come from:

`score[i, j] = x1[i]ᵀ · W_QK^h · x1[j]`

- **query side** = `x1[i]`, the stream at the destination
- **key side** = `x1[j]`, the stream at the source

In layer 2 each side is "token embedding + whatever layer-1 heads moved there", so multiplying out gives four terms: no composition (the one-layer QK circuit), Q-composition, K-composition, and both. Each is written `A_q ⊗ A_k ⊗ W`: slot 1 says whether query-side info moved before the dot product, slot 2 the same for the key side, slot 3 is a vocab × vocab table of scores. That's also where the "6-dimensional tensor" comes from: two `[n_context, d_model]` inputs and one `[n_context, n_context]` output.

### Induction heads

My first reading was "induction heads look for earlier copies of the current token". That's half of it. The pattern is `[a][b] … [a] → [b]`: find the earlier `a`, attend to the token **after** it, copy that.

Two heads, one circuit (term 3, K-composition):

1. **Previous-token head (layer 0 in the figures):** every position attends to the one before it and writes "the token before me was X" into a free subspace. After this, "Potter" in "Harry Potter" also carries "the token before me was Harry".
2. **Induction head (layer 1):** at the second "Harry", the query says "I am Harry". Every key has been relabelled to describe the *previous* token, so the key at "Potter" matches. The head attends to "Potter" and its OV circuit copies it. Prediction: Potter.

**Why it needs two layers.** In one layer, the key at position j can only be built from the token at j. To make it say something about j−1, an earlier head has to move that information in first.

Naming trap: equations count layers from 1 (H1, H2), figures count from 0. "1:8" means the second layer, head 8.

### Why the paper's evidence counts

Three levels, each stronger:

1. **Natural text (Harry Potter):** suggestive only, and the authors say so.
2. **Repeated random tokens:** random tokens have no statistics, so no memorized table can predict the second copy. Only the general "find it, copy what followed" rule works.
3. **The weights alone, no data:** the theory predicts the induction head's OV circuit is a copying matrix and its K-composition term with the previous-token head is a same-token matcher. They check both with the eigenvalue statistic `Σλ / Σ|λ|` (+1 all positive, −1 all negative, ~0 random). The induction heads sit near (+1, +1).

The circularity worry: they picked heads *because* K-composition was large, then checked K-composition. But "large" says nothing about sign. Both circuits coming out strongly positive was a prediction that could have failed. (The composition figure had a bug; the paper's correction keeps the main finding.)

---

## Where I am with this

- I understand the induction mechanism and what my plots and ablations need to show.
- The two-layer maths (tensor notation, path expansion of attention scores) I can follow on a re-read but can't fluently/intuitively reproduce yet.

## What this means for the project

1. Train the 2-layer attention-only model on repeated random-token sequences; watch the loss on the repeated half drop.
2. Plot every head's attention pattern on one repeated sequence; find the previous-token head (L0) and the induction head (L1).
3. Score them: previous-token score = average attention one position back; induction score = average attention to the position after the earlier twin.
4. Ablate. Induction head off → loss on the repeated half goes back up. Previous-token head off → the induction stripe disappears. That second one is the key result: two heads, one circuit. The attention plots only show where each head looks; the ablations show why.

Optional extras after the core result: composition scores and the two eigenvalue checks on my own weights.

## Questions I wrote down while reading

My raw questions, in paper order. Answered ones point to the section where I worked them out; Open = parked until after the project.

**Residual stream and virtual weights**
- Why do heads store info in directions perpendicular to what's already there? It has no effect on the other information because it's orthogonal. This is so clever. Is it an emergent behaviour we're observing? Partly answered (§2): nobody designs the directions; writers and readers drift into agreement through backprop. Open: how directly has this been observed?
- The token embeddings and the unembedding mostly interact with a small fraction of the dimensions. Why? Open: "it leaves room for the layers" is a plausible story, not something I've seen shown.
- A single MLP layer is 4x the residual stream width. That's crazy! And what are bottleneck activations? Open.
- Some MLP neurons and heads seem to do "memory management", clearing dimensions by writing the negative of what they read. I'd have expected this from heads, not MLP neurons. Does ReLU have a role here? Open.

**Attention heads**
- The paper prefers the concatenated definition of multi-head attention because it's a bigger, more efficient matrix multiply. But isn't it better to have lots of heads running independently, for flexibility? Or is it just saying the maths is equivalent? Open.
- "Attention heads typically write to a different subspace than they read from." My first guess was that position n+1 may depend on position n. Answered (§2, §4): the better reason is that what a component reads and what it writes mean different things. If it wrote back into the directions it read from, it could only amplify or dampen what was already there, never pass along a new kind of information.
- In "compute the result vector by combining value vectors", why `v_j` and not `v_i`, when the step before uses `v_i`? Open.
- Is `Id ⊗ W` like a matrix multiply and `A ⊗ Id` like an element-wise multiply? Answered (§6): both are matrix multiplies, along different axes.
- "But we can do it all in one step without referring to keys and queries." You what? This is bananas. And why would implementing attention that way be horribly inefficient? Open.
- The residual stream, often read as a "contextual word embedding", has subspaces holding information copied from *other* tokens. Answered (§7): after the previous-token head, "Potter" also carries "the token before me was Harry".
- What does fixing the attention pattern mean, and why are heads "half-linear" without it? Answered (§6).
- Keys, queries and values are by-products of the low-rank matrices and could be reparametrized without changing the head. Fair. How do we do that? Open.

**Zero- and one-layer models**
- The direct path represents a "residual" of bigram statistics in bigger models, e.g. "Barack" → "Obama". Not sure I fully get this. Open.
- Why is the transformer written as a product of three terms in reverse order? Answered (§6): function composition; read right to left, it's the forward pass.
- The direct path doesn't move information between positions "because that's what `Id ⊗ …` denotes". My guess: Id means no attention head in that term, so nothing moves between positions. Answered (§7): right.
- If multiple destination tokens attend to the same source token by the same amount, the source has the same effect on the logits. So if positions 5 and 7 both attend to position 2 with the same percentage, position 2 affects them the same way. Answered: makes sense, OV only ever sees the source token (§4).
- How does freezing make the logits linear in the tokens? Is it tokens → QK → [freeze] → OV → logits? Answered (§6): yes.
- What does "fungibility between variables" mean here? Open.
- QK weights aren't comparable between different query vectors, with no clear way to normalize them. Not sure what the second half means. Open.
- "The OV circuit sets things up so that tokens, if attended to, increase the probability of that token, and to a lesser extent, similar tokens." What? Answered (§7): that's copying.
- "Most of these examples are copying." Copying meaning the output token is the same as the source token? Answered: yes.
- In the split-token case ("R" → "alph"), how can "R" attend to "alph"? Attention can't move forwards. Maybe it's looking at earlier tokens where the word appeared? Answered (§6): that's it; it attends back to an earlier " Ralph".
- Python `else`/`elif`/`except` being more likely "after an indentation is reduced". What does reduced mean? Open.
- With standard positional embeddings, the one-layer maths "works out to multiplying W_QK by the positional embeddings". Open.
- Primarily positional attention heads. Open.
- The skip-trigram bugs section. This is interesting; want to dig in. Open.

**Eigenvalues and copying**
- How do eigenvectors and eigenvalues map onto OV/QK circuits? (Refresher needed.) Partly answered (§7).
- Why does copying require positive eigenvalues? Partly answered (§7): λ > 0 means a mix of tokens promotes itself.
- Matrices with all positive eigenvalues can still map some tokens to *decreasing* their own logit, because eigenvectors aren't necessarily orthogonal. I get maybe 85–90% of this. Open.
- "The sum of the eigenvalues is equal to the trace is equal to the sum of the diagonal." This feels like kind of a bombshell sentence. Open.

**Two-layer models**
- What are "virtual attention heads"? Answered (§7).
- Path expansion of logits: individual vs virtual heads in the second equation. Answered (§7).
- What are the query side and key side? Is it tensor notation? Answered (§7): no, it's where the two inputs to the dot product come from.
- What am I seeing in the interactive diagram? Do induction heads look for earlier copies of the present token? Answered (§7): half right; they attend to the token *after* the earlier copy.
- How do the diagrams show their theory is right? Answered (§7).
- Honestly, the whole two-layer section was fuzzy: I got what induction heads are, but not the maths before it or what the diagrams show and prove. Mostly answered now; the maths I can follow but not reproduce.

**Still to do:** I grasped the tensor notation only at a superficial level. Work through it properly, step by step.

## Why I care

What got me about this paper is the same thing that got me in *Zoom In* and in PiELo: two heads that individually do almost nothing (one copies the previous token, one copies whatever it attends to) compose into an algorithm, in-context copying, that neither has on its own. That's the question I keep coming back to: how simple units give rise to something that looks like understanding.
