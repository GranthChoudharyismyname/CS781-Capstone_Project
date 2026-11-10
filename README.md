# CS781 Capstone Project

This repository contains my implementation for the CS781 capstone
project. I approached the project as a sequence of related NLP tasks
built around a common legal-domain language model. Instead of training a
separate model from scratch for every task, I first trained a
decoder-only language model on legal text and then reused that model as
the backbone for the downstream phases.

The repository is organised around seven task-specific phases, along with
a separate notebook for training the custom byte-level BPE tokenizer:

1.  **Tokenizer** -- build a byte-level BPE tokenizer for the legal text
    corpus.
2.  **Phase 1 -- Legal Language Modelling** -- pretrain a decoder-only
    Transformer and use it for legal text generation.
3.  **Phase 2 -- Legal Named Entity Recognition (LNER)** -- adapt the
    Phase 1 model to identify legal entities as character-level spans.
4.  **Phase 3 -- Sentence-level Role Classification** -- use the Phase 1
    decoder to classify the role of each sentence in a document.
5.  **Phase 4 -- Document-level Classification (CJPE)** -- reuse the
    Phase 1 decoder to classify an entire legal document.
6.  **Phase 5 -- Legal Statute Identification (LSI)** -- predict the
    statutes relevant to a legal document using a multi-label classifier.
7.  **Phase 6 -- Legal Document Summarisation** -- generate summaries
    through a two-stage, chunk-based summarisation pipeline.
8.  **Phase 7 -- Precedent Case Retrieval (PCR)** -- retrieve relevant
    precedent cases for a given query case.

The main idea connecting the phases is transfer learning: the first
phase learns contextual representations from legal text, while the later
phases adapt those representations to different prediction tasks.

------------------------------------------------------------------------

## Repository contents

The notebooks shared for this implementation are:

  -----------------------------------------------------------------------
  Notebook                            Purpose
  ----------------------------------- -----------------------------------
  `bpe-tokenizer.ipynb`               Implements and trains the custom
                                      byte-level BPE tokenizer.

  `phase1.ipynb`                      Builds, trains and evaluates the
                                      decoder-only legal language model;
                                      generates the Phase 1 predictions.

  `phase2.ipynb`                      Performs legal named entity
                                      recognition using the Phase 1 model
                                      as the pretrained backbone.

  `phase3.ipynb`                      Performs sentence-level role
                                      classification, including LoRMA
                                      adaptation and knowledge
                                      distillation experiments.

  `phase4.ipynb`                      Performs document-level binary
                                      classification using chunk
                                      representations and attention
                                      pooling.
  -----------------------------------------------------------------------

Some notebooks create helper Python modules during execution using
notebook cells (for example, `attention.py`, `model_arch.py`, and the
`cjpe_*.py` modules in Phase 4). These modules are generated as part of
the notebook workflow rather than being assumed to exist independently
before the relevant cells run.

## Overall architecture

The high-level flow of the project is:

``` mermaid
flowchart TD
    A[Legal text datasets] --> B[Custom byte-level BPE tokenizer]
    B --> C[Phase 1: Decoder-only Transformer]
    C --> D[Pretrained legal language model]
    D --> E[Phase 2: Legal NER]
    D --> F[Phase 3: Sentence role classification]
    D --> G[Phase 4: Document classification]
    D --> K[Phase 5: Statute identification]
    D --> L[Phase 6: Document summarisation]
    D --> M[Phase 7: Precedent retrieval]
    E --> H[Entity spans]
    F --> I[Sentence-level role labels]
    G --> J[Document-level binary prediction]
    K --> N[Relevant statute IDs]
    L --> O[Generated summaries]
    M --> P[Top-k precedent cases]
```

The same tokenizer and Phase 1 decoder are reused in the downstream
tasks. Each downstream phase adds a task-specific prediction head and a
task-specific data pipeline.

------------------------------------------------------------------------

# 1. Custom Byte-level BPE Tokenizer

**Notebook:** `bpe-tokenizer.ipynb`

Before training the language model, I implemented a custom byte-level
Byte Pair Encoding (BPE) tokenizer. The tokenizer has a target
vocabulary size of 30,000 tokens and reserves four token IDs:

  Token       ID Purpose
  --------- ---- ---------------------------------------
  `<pad>`      0 Padding shorter sequences in a batch.
  `<unk>`      1 Unknown-token fallback.
  `<s>`        2 Beginning-of-sequence marker.
  `</s>`       3 End-of-sequence marker.

### 1.1 Why byte-level tokenization?

Legal text contains a mixture of ordinary words, names, citations,
abbreviations, punctuation, numbers and sometimes unusual symbols. A
tokenizer that depends only on a fixed character or word vocabulary can
struggle with rare strings.

I therefore represent the UTF-8 bytes of each word using a reversible
byte-to-Unicode mapping. Every possible byte value has a corresponding
symbol. This means that the initial representation can cover arbitrary
text without requiring a separate unknown token for every unseen word.

An end-of-word marker, `</w>`, is appended to each word. It allows the
learned merges to retain information about word boundaries.

### 1.2 BPE training

The training procedure starts with:

-   The four reserved tokens.
-   The 256 byte symbols.
-   The end-of-word symbol.

I then split the training corpus on whitespace and convert each word
into its sequence of byte symbols followed by `</w>`. Repeated words are
stored with their frequencies so that the pair-counting process can use
word frequency rather than repeatedly processing identical word
sequences.

At each BPE iteration:

1.  Count adjacent symbol pairs across the current word representations,
    weighted by word frequency.
2.  Find the pair with the highest frequency.
3.  Resolve ties deterministically by sorting the corresponding symbols.
4.  Merge the selected pair into a new symbol.
5.  Update the affected word representations and pair counts.
6.  Add the new symbol to the vocabulary and record the merge order.

This is frequency-based BPE: the merge criterion in the implementation
is the count of an adjacent pair. It is not an NPMI- or IDF-based merge
score.

The process continues until the vocabulary reaches its target size or
there are no remaining pairs with frequency at least two.

### 1.3 Encoding and decoding

For encoding, each word is converted to byte symbols and receives the
end-of-word marker. The tokenizer repeatedly selects the available
adjacent pair with the lowest learned merge rank and merges it. The
resulting symbols are mapped to token IDs.

For decoding, token IDs are mapped back to their symbols, the byte
representation is reconstructed, and the bytes are decoded as UTF-8. The
end-of-word marker is used to restore spaces between words.

The notebook also implements saving and loading the vocabulary and merge
list, and pushing the tokenizer files to the Hugging Face Hub. The
downstream phases load the saved `vocab.json` and `merges.txt` files so
that their token IDs remain compatible with the Phase 1 model.

------------------------------------------------------------------------

# 2. Phase 1 -- Legal Language Modelling

**Notebook:** `phase1.ipynb`

The first phase is the foundation for the rest of the project. I trained
a decoder-only Transformer language model on legal text so that it could
learn legal vocabulary, phrasing, contextual relationships and document
patterns.

## 2.1 Training data

The Phase 1 notebook uses four legal text sources from the project
dataset repository:

  Source            Sampling proportion
  --------------- ---------------------
  Acts                              10%
  Supreme Court                     40%
  High Courts                       40%
  Tribunals                         10%

The notebook configures a total token budget of 400 million tokens,
including a 2-million-token validation allocation. Training examples are
packed into fixed-length blocks of 1,024 tokens.

The data pipeline streams the JSON data, tokenizes the text, separates
the training and validation token streams, and writes token blocks to
disk. A block dataset then loads these blocks for language-model
training.

## 2.2 Transformer architecture

The language model is a decoder-only Transformer implemented in PyTorch.
The Phase 1 configuration is:

  Component                                          Configuration
  ------------------------ ---------------------------------------
  Vocabulary size            Custom BPE vocabulary (30,000 target)
  Context length                                      1,024 tokens
  Transformer layers                                            12
  Model dimension                                              768
  Attention heads                                               12
  Head dimension                                                64
  Feed-forward dimension                                     3,072
  Dropout                                                        0

The model can be understood as the following sequence:

``` mermaid
flowchart TD
    A[Input token IDs] --> B[Token embedding]
    B --> C[Transformer block 1]
    C --> D[Transformer block 2]
    D --> E[...]
    E --> F[Transformer block 12]
    F --> G[Final LayerNorm]
    G --> H[Language-model output projection]
    H --> I[Next-token logits]
```

The input embedding maps each token ID to a 768-dimensional vector. The
output projection maps the final hidden representation back to
vocabulary-sized logits. The output projection shares its weights with
the token embedding matrix, so the model uses tied input and output
embeddings.

Each Transformer block follows a pre-LayerNorm residual structure:

``` mermaid
flowchart TD
    A[Hidden state x] --> B[LayerNorm]
    B --> C[Causal self-attention]
    C --> D[Residual addition]
    A --> D
    D --> E[LayerNorm]
    E --> F[Feed-forward network]
    F --> G[Residual addition]
    D --> G
    G --> H[Output hidden state]
```

The feed-forward network consists of a linear projection from 768 to
3,072 dimensions, GELU activation, dropout, and a projection back to 768
dimensions.

## 2.3 Causal self-attention and RoPE

For every attention head, the hidden states are projected into queries,
keys and values. Attention is causal: a token can attend to itself and
earlier tokens, but not to future tokens. This is what allows the model
to predict the next token without seeing the answer in advance.

The attention score is scaled by the inverse square root of the head
dimension:

\[ `\operatorname{Attention}`{=tex}(Q,K,V) =
`\operatorname{softmax}`{=tex} `\left`{=tex}(
`\frac{QK^\top}{\sqrt{d_{\text{head}}}}`{=tex} + M `\right`{=tex})V \]

where (M) is the causal mask.

The implementation uses Rotary Positional Embeddings (RoPE) on the query
and key representations. RoPE rotates pairs of dimensions according to
their token positions. This injects positional information into
attention while allowing the attention score to encode relative-position
information.

The attention module also contains a learned gate:

\[ g = `\sigma`{=tex}(W_g x) \]

\[ h\_{`\text{attn}`{=tex}} =
W_o`\left`{=tex}(`\operatorname{Attention}`{=tex}(Q,K,V)`\odot `{=tex}g`\right`{=tex})
\]

Here, (W_g) is the gate projection, (`\sigma`{=tex}) is the sigmoid
function, and (`\odot`{=tex}) is element-wise multiplication. The gate
is applied to the attention output before the output projection. It lets
the block modulate how much of the attention result is passed forward at
each hidden dimension.

The implementation uses PyTorch scaled dot-product attention for the
ordinary causal-attention path and an explicit masked-attention path
when padding masks or cached key/value states are involved.

## 2.4 Language-modelling objective and training

The model is trained using the causal next-token prediction objective.
Given a sequence

\[ x_1,x_2,`\ldots`{=tex},x_T, \]

the model learns to predict

\[ x_2,x_3,`\ldots`{=tex},x\_{T+1} \]

from the preceding context. The training loss is cross-entropy between
the predicted vocabulary distribution and the next-token target. Padding
or ignored positions are excluded from the loss.

The notebook uses the Hugging Face `Trainer` interface with the custom
model and a custom trainer implementation for saving the tied-weight
model correctly. The configured training setup includes:

-   One training epoch.
-   Per-device training batch size of 4.
-   Gradient accumulation of 2 steps.
-   Learning rate of (3`\times10`{=tex}\^{-4}).
-   Weight decay of 0.1.
-   500 warm-up steps and a cosine learning-rate schedule.
-   Mixed-precision training when CUDA is available.
-   Periodic logging, evaluation and checkpoint saving.

Validation loss is also used to calculate perplexity:

\[ `\operatorname{Perplexity}`{=tex} =
`\exp`{=tex}(`\operatorname{CrossEntropyLoss}`{=tex}) \]

Perplexity gives an interpretable measure of how uncertain the model is
when predicting the next token. Lower perplexity indicates better
next-token prediction on the evaluated text.

## 2.5 Text generation

After training, the model is saved and uploaded to the Hugging Face Hub.
The generation pipeline loads the trained checkpoint and the matching
tokenizer.

The notebook implements both ordinary generation and beam-search
generation. For the Phase 1 prediction pipeline, prompts are tokenized
and padded, and the model generates continuations using beam search. The
configured generation parameters include:

-   4 beams.
-   Temperature of 0.3.
-   Maximum generation budget of 384 tokens.
-   A no-repeat 3-gram constraint.
-   A target of 128 generated words.

The model uses a key/value cache during autoregressive generation.
Instead of recomputing the keys and values for the entire prompt at
every step, it reuses the cached representations and processes the newly
generated token. This reduces redundant computation during generation.

The generated predictions are decoded, formatted into the expected JSON
structure and packaged into a ZIP file for submission.

------------------------------------------------------------------------

# 3. Phase 2 -- Legal Named Entity Recognition

**Notebook:** `phase2.ipynb`

In Phase 2, I reuse the pretrained Phase 1 decoder and adapt it for
legal named entity recognition. The task is different from language
modelling: instead of predicting the next token, the model must identify
entity spans in the original text and assign an entity type to each
span.

## 3.1 Dataset and label representation

The notebook loads the Phase 2 LNER text and gold annotations. The gold
annotations are represented using character offsets:

\[ (start, end, label) \]

I convert these character-level spans into token-level BIO tags:

-   `B-<LABEL>` marks the beginning of an entity.
-   `I-<LABEL>` marks a continuation token inside that entity.
-   `O` marks a token outside any entity.

Positions with empty offsets are assigned the ignore index `-100`, so
they do not contribute to the token-classification loss.

## 3.2 Token-to-character alignment

A major part of this phase is preserving the relationship between the
tokenizer's output and the original text. The tokenizer works on
byte-level symbols, while the annotations use character offsets.

I therefore implement an offset-aware encoding pipeline:

1.  Split punctuation from the beginning and end of words where
    required.
2.  Encode each resulting word segment using the same BPE tokenizer as
    Phase 1.
3.  Track the UTF-8 byte length represented by each BPE piece.
4.  Map byte positions back to Python character positions.
5.  Store a character offset for every generated token.

The gold entity spans can then be projected onto the token sequence. At
inference time, the predicted BIO tags are decoded back into character
spans. This is important because the final output must refer to
positions in the original document, not merely token indices.

## 3.3 Long-document windowing

The maximum token window is 512 tokens, with a stride/overlap setting of
128 tokens. Long documents are divided into overlapping windows so that
the model can process them without exceeding the sequence-length limit.

During inference, I use a window plan that assigns central regions of
overlapping windows to the final prediction. The overlap provides
context near window boundaries, while the ownership regions prevent the
same token from being predicted multiple times.

## 3.4 Model architecture

The Phase 2 model consists of the pretrained decoder, a bidirectional
GRU and a token-classification layer:

``` mermaid
flowchart TD
    A[Input token IDs] --> B[Phase 1 token embeddings]
    B --> C[12-layer pretrained decoder]
    C --> D[Final LayerNorm]
    D --> E[Dropout]
    E --> F[Bidirectional GRU]
    F --> G[Dropout]
    G --> H[Linear token-classification head]
    H --> I[BIO tag logits]
    I --> J[Decode tags into character spans]
```

For each input token, the decoder produces a contextual hidden
representation of size 768. These contextual representations are passed
through a bidirectional GRU with a hidden size of 256 in each direction.
The concatenated forward and backward GRU states are mapped to the BIO
tag vocabulary using a linear classification layer.

The GRU adds another sequence-modelling layer on top of the Transformer
representations. The Transformer provides contextual token features,
while the bidirectional GRU learns additional local sequence
relationships useful for deciding where an entity begins and ends.

## 3.5 Fine-tuning strategies

The notebook defines multiple ways to adapt the pretrained backbone:

-   **Full fine-tuning:** update all parameters in the pretrained
    decoder.
-   **Top-layer fine-tuning:** freeze most of the decoder and update
    only the selected final Transformer blocks and final LayerNorm.
-   **LoRA:** freeze the original weights and add trainable low-rank
    adapters to selected attention projections.
-   **LoRMA:** use the notebook's custom low-rank adaptation wrapper for
    selected linear projections.

These alternatives make it possible to compare the amount of trainable
parameters and the adaptation behaviour. The final configuration
selected in the notebook is full fine-tuning, with a backbone learning
rate of (5`\times10`{=tex}\^{-5}), a new-layer learning rate of
(5`\times10`{=tex}\^{-4}), and 15 epochs.

The token-classification objective is cross-entropy over the BIO labels.
The implementation supports class weighting to reduce the effect of
label imbalance and ignores positions marked `-100`. It uses a cosine
learning-rate schedule with warm-up, gradient clipping and mixed
precision when CUDA is available.

## 3.6 Inference and submission

At inference time, the model predicts token-level BIO tags for each
window. The predictions are stitched back into document order, decoded
into character spans and optionally adjusted to the beginning of a word.
The final predictions are saved in `prediction.json` and packaged into a
ZIP file.

------------------------------------------------------------------------

# 4. Phase 3 -- Sentence-level Role Classification

**Notebook:** `phase3.ipynb`

Phase 3 uses the Phase 1 language model for sentence-level role
classification. The input is a document represented as a sequence of
sentences, and the model predicts a role label for every sentence.

The notebook loads the CL and IT dataset splits, combines the
corresponding training and development documents, and uses the provided
label vocabulary to convert role names into integer IDs.

## 4.1 Sentence representation from the decoder

Each sentence is tokenized and passed independently through the
pretrained Phase 1 decoder. The decoder produces a contextual hidden
state for every token.

Rather than using only one token representation, I construct a sentence
vector using two summaries:

1.  **Mean pooling:** average the final hidden states over the
    non-padding tokens.
2.  **Last-token representation:** take the hidden state of the final
    non-padding token.

These two vectors are concatenated:

\[ s_i =
\[`\operatorname{MeanPool}`{=tex}(H_i); H\_{i,`\text{last}`{=tex}}\] \]

Since the decoder dimension is 768, each sentence is represented by a
1,536-dimensional vector.

This combines information distributed across the sentence with the
representation at its final token.

## 4.2 BiGRU over the document

The sentence vectors are grouped back into their original document
sequences. A LayerNorm and dropout are applied before the sequence
model.

The model then uses a bidirectional GRU with a hidden size of 256 in
each direction. The GRU processes the sequence of sentence vectors,
allowing the prediction for a sentence to use information from both
earlier and later sentences.

``` mermaid
flowchart TD
    A[Document: sentence 1 ... sentence N] --> B[Tokenize each sentence]
    B --> C[Shared Phase 1 decoder]
    C --> D[Final token hidden states]
    D --> E[Mean pooling]
    D --> F[Last-token state]
    E --> G[Concatenate sentence features]
    F --> G
    G --> H[LayerNorm + Dropout]
    H --> I[Bidirectional GRU over sentences]
    I --> J[Linear role classifier]
    J --> K[One role label per sentence]
```

The final linear layer maps the bidirectional GRU output to the number
of role labels. Thus, the classification is performed at the sentence
level, but each sentence prediction can incorporate document-level
sequential context through the BiGRU.

The implementation also supports running the GRU over fixed-size
sentence windows. During evaluation, the notebook compares
whole-document processing with windowed processing.

## 4.3 LoRMA adaptation

For this phase, I implement a custom low-rank adaptation module called
`LoRMALinear`. The Phase 1 decoder is frozen, and the selected attention
projections are wrapped with trainable low-rank parameters.

The configured targets are the query and value projections:

-   `attn.q_proj`
-   `attn.v_proj`

The rank is 8 and the scaling parameter is 16. The selected variant is
`plus`.

The adapter computes a low-rank update to the frozen linear
transformation. In the `plus` variant, the update is applied as a
residual transformation of the base projection output. The code also
supports merging the adapter into ordinary linear weights after
training, so that inference does not need to retain the separate adapter
computation.

## 4.4 Teacher/student training and knowledge distillation

The notebook includes a teacher/student training path. The teacher is
built from the full 12-layer Phase 1 decoder. The student is built by
selecting decoder layers from the pretrained model according to evenly
spaced layer indices.

The student is trained using a combination of:

-   **Hard-label loss:** cross-entropy against the ground-truth sentence
    roles.
-   **Soft-label loss:** KL divergence between the student and teacher
    output distributions.
-   **Feature loss:** cosine-distance-based alignment between student
    and teacher sentence representations.

The combined objective is:

\[ `\mathcal{L}`{=tex} =
`\alpha `{=tex}`\mathcal{L}`{=tex}*{`\text{hard}`{=tex}} +
(1-`\alpha`{=tex})`\mathcal{L}`{=tex}*{`\text{soft}`{=tex}} +
`\lambda `{=tex}`\mathcal{L}`{=tex}\_{`\text{feature}`{=tex}} \]

The soft-label term uses temperature scaling. The notebook configures a
temperature of 2, a hard-loss coefficient of 0.5 and a feature-loss
weight of 0.5.

One detail worth noting is that the notebook's default student depth is
also configured as 12 layers. With that setting, the selected student
layer indices cover all 12 teacher layers; the code supports a
reduced-depth student if `num_student_layers` is changed, but the
default configuration does not reduce the decoder depth.

The teacher is trained and evaluated first. If knowledge distillation is
enabled, the student is then trained using the teacher's logits and
sentence representations. The notebook uses the student for submission
in that configuration.

## 4.5 Output

The model predicts one role label for each sentence. The predictions are
converted from integer IDs back to their label names, stored by document
ID in `prediction.json`, and packaged into a ZIP file.

The notebook also saves the merged decoder weights, the task-specific
head weights and the selected decoder-layer indices.

------------------------------------------------------------------------

# 5. Phase 4 -- Document-level Classification (CJPE)

**Notebook:** `phase4.ipynb`

Phase 4 moves from sentence-level predictions to a document-level binary
classification task. The input is a complete legal document, and the
output is one of two classes.

The main challenge is document length. The pretrained decoder has a
context length of 1,024 tokens, while a legal judgment can be much
longer. I handle this by splitting each document into chunks, encoding
the chunks independently and then combining their representations using
attention pooling.

## 5.1 Data pipeline and chunking

The notebook reads the train, development and test JSON files from the
`Phase_4_CJPE` dataset directory. It supports streaming JSON records and
caches tokenized documents to disk so that the tokenizer does not need
to process the same documents again on every run.

The configured chunking parameters are:

  Parameter                            Value
  ----------------------------- ------------
  Chunk length                    512 tokens
  Chunk overlap                    64 tokens
  Maximum chunks per document             16
  Training fraction                      50%
  Batch size                               4
  Gradient accumulation                    4

For a long document, the chunking function creates windows of 512 tokens
with an overlap of 64 tokens. If a document would produce more than 16
chunks, the implementation selects chunk positions distributed across
the document. This keeps the computation bounded while retaining
information from different parts of the document.

The chunks belonging to a document are kept grouped together so that
their representations can later be aggregated into one document
representation.

## 5.2 Chunk encoder

Every chunk is passed through the pretrained Phase 1 decoder. The
decoder is reused as the backbone, with the same 12-layer,
768-dimensional, 12-head configuration.

For each chunk, I take the final hidden states and compute a masked mean
over the non-padding tokens:

\[ h_j = `\frac{\sum_t m_{jt} H_{jt}}`{=tex}
{`\max`{=tex}(`\sum`{=tex}*t m*{jt}, 1)} \]

Here, (H\_{jt}) is the hidden state at token position (t) in chunk (j),
and (m\_{jt}) is its attention/padding mask.

This produces one 768-dimensional vector for every chunk.

To control memory usage, chunks are encoded in smaller groups. The
implementation also supports gradient checkpointing during training,
which trades additional computation for lower activation-memory usage.

## 5.3 Attention pooling over chunks

A document can contain many chunks, and not every chunk is equally
useful for the final classification. Instead of simply averaging all
chunk vectors, I use a learned attention-pooling module.

For every chunk representation (h_j), the pooling layer calculates a
score:

\[ e_j = w\^`\top `{=tex}`\tanh`{=tex}(W h_j) \]

The scores are normalized across the valid chunks:

\[ a_j = `\frac{\exp(e_j)}{\sum_k \exp(e_k)}`{=tex} \]

The final document representation is the weighted sum:

\[ h\_{`\text{doc}`{=tex}} = `\sum`{=tex}\_j a_j h_j \]

The mask ensures that padded chunk positions do not receive attention.

``` mermaid
flowchart TD
    A[Legal document] --> B[Tokenizer]
    B --> C[512-token chunks with overlap]
    C --> D[Pretrained Phase 1 decoder]
    D --> E[Masked mean pooling per chunk]
    E --> F[Chunk representations]
    F --> G[Learned attention scores]
    G --> H[Softmax over valid chunks]
    H --> I[Weighted document representation]
    I --> J[Dropout]
    J --> K[Linear layer: 2 logits]
    K --> L[Document class]
```

This is a hierarchical approach: the Transformer builds contextual
representations inside each chunk, and the attention-pooling layer
combines the chunk representations into a document-level representation.

## 5.4 Classification head and training

The document representation is passed through dropout and a linear layer
that outputs two logits.

The loss is weighted cross-entropy. The class weights are computed from
the training-label distribution so that the less frequent class receives
a larger contribution to the loss.

The notebook's configured training setup is:

-   Full fine-tuning of the Phase 1 backbone.
-   Three epochs.
-   Backbone learning rate of (3`\times10`{=tex}\^{-5}).
-   Classification-head learning rate of (5`\times10`{=tex}\^{-4}).
-   Weight decay of 0.01.
-   Gradient accumulation of 4 steps.
-   Cosine learning-rate schedule with warm-up.
-   Gradient clipping.
-   Mixed precision when CUDA is available.

The optimizer uses separate parameter groups for the backbone and the
newly introduced pooling/classification head. The head has a higher
learning rate because it is newly initialized, while the pretrained
backbone is updated more conservatively.

## 5.5 Evaluation and prediction threshold

The notebook evaluates the model using macro precision, macro recall and
macro-F1. It also searches over probability thresholds from 0.05 to 0.95
on the development set and selects the threshold that gives the best
development macro-F1.

The selected threshold is then used to convert the positive-class
probability into the final binary prediction.

The notebook also records model size, number of trainable parameters,
throughput and peak GPU memory. It calculates an attention profile over
document positions to inspect how much attention the pooling layer
assigns to the beginning, middle and end of documents.

The final predictions are written as JSON records containing the
document ID and predicted class, then packaged into a ZIP file.

------------------------------------------------------------------------


------------------------------------------------------------------------

# 6. Phase 5 -- Legal Statute Identification (LSI)

**Notebook:** `phase5.ipynb`

Phase 5 predicts the statutes relevant to a legal document. Since a
document can be associated with more than one statute, this is treated as
a multi-label classification task rather than ordinary single-class
classification.

The notebook loads the Phase 5 train, development and test splits, converts
the statute IDs into multi-hot target vectors, and uses the Phase 1
decoder as its pretrained backbone. Documents are tokenized and divided
into chunks of 1,024 tokens. At most four chunks are retained, sampled at
evenly spaced positions when a document is longer.

The model applies masked mean pooling to the token representations of
each chunk and averages the chunk vectors to obtain one document
representation. A linear head produces two logits for every statute,
representing the negative and positive class. The model is trained with
one-hot cross-entropy over these per-statute predictions.

For parameter-efficient adaptation, the notebook applies custom LoRMA
adapters to the query, key, value and output attention projections. The
backbone weights remain frozen, while the adapters and classification
head are trained. The configured run uses rank 16, alpha 32, six epochs,
a batch size of 2 and gradient accumulation over four steps. Training
uses 10% of the available training documents, with separate learning
rates for the adapters and classification head.

At evaluation time, the notebook reports macro-F1, micro-F1 and subset
accuracy. It searches a threshold grid on the development set to select
the threshold with the best macro-F1. If a document receives no positive
prediction at that threshold, its highest-scoring statute is retained.
The final output contains each document ID and its predicted statute IDs.

------------------------------------------------------------------------

# 7. Phase 6 -- Legal Document Summarisation

**Notebook:** `phase-6-next(1).ipynb`

Phase 6 generates a summary for each legal document. Since judgments can
be much longer than the decoder's 1,024-token context, the notebook uses
a two-stage approach: first summarising individual semantic chunks, and
then combining those summaries into a final document-level summary.

Documents are split into sentences and then into semantic chunks. TF-IDF
features are used to estimate topic changes between neighbouring
sentences, while token-budget limits and sentence overlap control chunk
length and preserve some context at boundaries. The training pipeline
aligns reference-summary sentences with the chunks they are most similar
to, creating chunk-level training examples for the first stage. It also
creates a second-stage example in which extractive chunk summaries are
combined and mapped to the full reference summary.

The model reuses the Phase 1 decoder and adds parameter-efficient
adaptation through LoRMA+. The adapters are attached to the attention
projections, attention gate and feed-forward projections, while the
original model weights remain frozen. Training uses a causal
language-modelling objective, with loss computed on the target summary
tokens rather than the prompt. The configured run uses rank 32, alpha
64, two epochs, a batch size of 4 and gradient accumulation over four
steps. It uses 10% of the training documents and evaluates validation
loss and perplexity while saving the best adapter checkpoint.

At inference, the model first generates summaries for the semantic chunks.
These are joined within the second-stage context budget and passed through
the model again to produce the final summary. The notebook evaluates
summaries using ROUGE-1, ROUGE-2 and ROUGE-L, with optional BERTScore
support. Predictions are saved with document IDs in `prediction.json`
and packaged for submission.

------------------------------------------------------------------------

# 8. Phase 7 -- Precedent Case Retrieval (PCR)

**Notebook:** `phase7.ipynb`

Phase 7 retrieves relevant precedent cases for a query case. Each query
is compared against a candidate pool, and the model returns the seven
highest-ranked candidate cases.

The notebook first builds a compact representation, or fingerprint, for
each case. It splits cases into sentences, encodes them using the Phase 1
decoder and detects topic boundaries from the similarity between
neighbouring sentence windows. Each resulting chunk is summarised, and
the summaries are progressively combined to form a final case
fingerprint. The code supports using an abstractive summariser from
Phase 6, but the configured notebook sets `summarizer_repo = None`;
therefore, it uses a centroid-based extractive summariser by default.
Fingerprints are cached so that completed cases do not need to be
processed again.

The retrieval model encodes each fingerprint with the pretrained
encoder and mean-pools its token representations into a normalised
case vector. It is adapted using custom AdaLoRA layers on the attention
projections and, by default, the feed-forward projections as well. The
initial adapter rank is 24 and the target average rank is 16. Training
uses a contrastive objective: each query is paired with a relevant case
and hard negative candidates selected using TF-IDF similarity. An
orthogonality penalty is added to the retrieval loss, and the adapter
rank budget is gradually reduced during training.

At evaluation, cosine similarity between query and candidate embeddings
is used to rank the candidate pool. The notebook also tests a weighted
combination of dense similarity and TF-IDF similarity, selecting the
keyword weight using development-set F1@7. It reports precision, recall
and F1 at seven retrieved cases, excludes the query itself when it is
present in the candidate pool, and writes the top-seven case IDs for
each test query to `prediction.json`.

------------------------------------------------------------------------

# 9. How the phases fit together

The project uses one pretrained language model as the shared
representation backbone, but the task-specific heads are different.

  -----------------------------------------------------------------------------
  Phase          Input unit     Shared         Task-specific     Output
                                backbone       component         
  -------------- -------------- -------------- ----------------- --------------
  Phase 1        Token sequence Decoder-only   Vocabulary        Next-token
                                Transformer    projection with   distribution /
                                trained from   tied embeddings   generated text
                                scratch                          

  Phase 2        Token sequence Phase 1        BiGRU + token     BIO tags
                 in windows     decoder        classifier        converted to
                                                                 entity spans

  Phase 3        Sentence       Phase 1        Mean/last-token   Role label per
                 sequence       decoder        pooling +         sentence
                                applied to     document BiGRU +  
                                each sentence  classifier        

  Phase 4        Document       Phase 1        Chunk attention   One class per
                 chunks         decoder        pooling + binary  document
                                applied to     classifier        
                                each chunk                       
  Phase 5        Document       Phase 1        Chunk mean pooling Relevant statute
                 chunks         decoder        + LoRMA +          IDs (multi-label)
                                applied to     classification
                                each chunk     head

  Phase 6        Semantic       Phase 1        LoRMA+ generation  Document summary
                 chunks         decoder        in two stages

  Phase 7        Case           Phase 1        Case fingerprints Top-seven
                 fingerprints   decoder        + AdaLoRA          precedent cases
                                encoder        retrieval
  -----------------------------------------------------------------------------

The important distinction is where the task-specific sequence modelling
happens:

-   Phase 2 predicts labels at the token level and uses a BiGRU over
    token representations.
-   Phase 3 first creates one vector per sentence and then uses a BiGRU
    over the sentence sequence.
-   Phase 4 first creates one vector per chunk and then uses attention
    pooling to create one vector for the entire document.
-   Phase 5 averages chunk representations into a document vector and
    predicts multiple relevant statutes.
-   Phase 6 generates chunk summaries first and then generates a final
    summary from the combined intermediate summaries.
-   Phase 7 creates a fingerprint for each case and learns to rank
    candidate cases using dense similarity, optionally combined with
    TF-IDF similarity.

This allows the same pretrained language model to support tasks with
very different output structures.

------------------------------------------------------------------------

# 10. Running the notebooks

The notebooks were written around a Kaggle-style execution environment
and use the Hugging Face Hub for datasets, tokenizer files and
pretrained model weights.

## 10.1 Requirements

The main libraries used across the notebooks include:

-   Python
-   PyTorch
-   NumPy
-   Transformers
-   Hugging Face Hub
-   scikit-learn
-   PEFT
-   ijson
-   pandas
-   matplotlib

The exact imports and any additional packages are present in the
corresponding notebooks. Some cells install packages or generate helper
modules during execution.

A CUDA-enabled GPU is recommended, particularly for Phase 1 pretraining
and the downstream fine-tuning phases.

## 10.2 Hugging Face access

The notebooks load data and model files from the Hugging Face Hub. In
the Kaggle environment, I read the Hugging Face access token from Kaggle
Secrets using the secret name `HF`. Where the notebook supports it, the
token can also be supplied through the `HF_TOKEN` environment variable.

Do not commit an access token to the repository.

The project uses these Hub repositories:

  ----------------------------------------------------------------------------------------
  Repository                                           Purpose
  ---------------------------------------------------- -----------------------------------
  `Exploration-Lab/CS-781-Capstone`                    Project datasets

  `Granth2/cs781-custom-bpe-30k-Version-10-40-40-10`   Custom tokenizer vocabulary and
                                                       merge files

  `Granth2/cs781-phase1-Submission-9`                  Phase 1 model checkpoint used by
                                                       downstream phases
  ----------------------------------------------------------------------------------------

## 10.3 Execution order

For a fresh run, the intended order is:

1.  Run `bpe-tokenizer.ipynb` to train and save the tokenizer, if the
    tokenizer files have not already been created.
2.  Run `phase1.ipynb` to train the language model and publish the
    checkpoint.
3.  Run `phase2.ipynb` to fine-tune the pretrained model for legal NER.
4.  Run `phase3.ipynb` to train the sentence-level role classifier.
5.  Run `phase4.ipynb` to train the document-level classifier.
6.  Run `phase5.ipynb` to train the statute-identification model.
7.  Run `phase-6-next(1).ipynb` to train the summarisation model.
8.  Run `phase7.ipynb` to build case fingerprints and train the retriever.

The downstream phases expect the tokenizer and Phase 1 checkpoint to be
available from the configured Hugging Face repositories. If those
artifacts already exist, the tokenizer-training and Phase 1 pretraining
steps do not need to be repeated.

------------------------------------------------------------------------

# 11. Implementation notes

A few implementation decisions are particularly important when
reproducing the project:

-   **Tokenizer/checkpoint compatibility:** the vocabulary size and
    token-to-ID mapping must match the Phase 1 embedding matrix. Phase 2
    explicitly checks that the tokenizer vocabulary size matches the
    checkpoint.
-   **Long inputs:** the downstream tasks use different strategies for
    long documents. Phase 2 uses overlapping token windows, Phase 3
    groups sentence representations into windows for sequence
    processing, and Phase 4 uses chunk-level encoding followed by document-level attention
    pooling, Phase 5 averages selected chunk representations, Phase 6
    uses semantic chunks in a two-stage generation pipeline, and Phase 7
    builds topic-based case fingerprints.
-   **Padding and ignored labels:** padding positions are masked or
    assigned an ignore index so that they do not incorrectly influence
    pooling or the training loss.
-   **Transfer learning:** the downstream phases reuse the Phase 1
    decoder rather than initializing a new language model for each task.
-   **Memory management:** mixed precision, gradient accumulation,
    chunked encoding and gradient checkpointing are used where
    appropriate to make training feasible on GPU hardware.
-   **Task-specific evaluation:** Phase 2 evaluates exact entity spans,
    Phase 3 evaluates sentence-role predictions using macro-F1, and
    Phase 4 evaluates document-level classification using macro
    precision, recall and F1; Phase 5 uses macro-F1, micro-F1 and subset
    accuracy; Phase 6 uses ROUGE and optionally BERTScore; and Phase 7
    evaluates retrieval using precision, recall and F1@7.

The notebooks contain the actual configuration values and execution
code. If a configuration is changed, the corresponding checkpoint,
tokenizer and prediction files should be kept consistent.

------------------------------------------------------------------------

# 12. Limitations and reproducibility

The notebooks depend on the dataset and model artifacts hosted on
Hugging Face, and some cells expect a Kaggle environment with GPU access
and configured secrets. Running the notebooks outside Kaggle may require
adapting the paths, secret loading and hardware settings.

The maximum number of chunks in Phase 4 is capped at 16, while Phase 5
retains at most four evenly spaced chunks per document. Phase 2 and Phase 3
also use windowing strategies, and Phase 6 trains on a configured 10% of
the training documents. In Phase 7, the default case-fingerprint pipeline
uses extractive summaries unless a Phase 6 summariser checkpoint is
configured. These are practical constraints of the implemented pipelines
and should be considered when interpreting model behaviour.

The README describes the architectures and configurations present in the
submitted notebooks. It does not assume a particular leaderboard score
or claim an experiment succeeded unless that result is recorded in the
notebook outputs.

------------------------------------------------------------------------

## Final note

The project was built around the idea that a legal-domain language model
can act as a reusable foundation for multiple legal NLP tasks. Phase 1
learns general contextual representations from legal text, while the
following phases adapt those representations to entity extraction,
sentence-role classification, document classification, statute
identification, summarisation and precedent retrieval.

The most important part of the implementation, for me, is not just the
Transformer itself, but how the same pretrained representation model can
be connected to different task-specific architectures depending on the
level at which a prediction is required: token, sentence, document or case level.
