# GPT-2 Knowledge Distillation Project

This project studies knowledge distillation for language models. A pretrained GPT-2 small model is used as the teacher, and smaller GPT-2-style student models are trained and compared.

## Project Goal

The goal is to compress GPT-2 into a smaller student model while keeping reasonable language-modeling quality. The project compares three student training methods:

1. **Teacher + corpus**: learns from both the real text corpus and the teacher's soft probability distribution.
2. **Corpus only**: learns only from the real next-token labels in the dataset.
3. **Teacher only**: learns only from the teacher's soft probability distribution.

The project also runs a **temperature sweep** before final training. Temperature controls how soft the teacher's output distribution is during distillation. We train a throwaway student at each of `T = 1, 2, 4, 7, 10, 15, 20, 30`, then keep the temperature with the best validation perplexity for the final students. This helps us learn how sensitive distillation is to temperature and whether softer teacher probabilities improve student learning.

The sweep runs are trained with the **teacher-only** objective. Temperature affects only `teacher_loss`, so if the sweep used the combined loss then half of every gradient would come from `corpus_loss` no matter what the temperature was, diluting the effect we are trying to measure. See [Temperature Sweep Design](#temperature-sweep-design).

## Research Questions

- Can a smaller GPT-2-style student retain useful language-modeling ability from GPT-2 small?
- Does teacher guidance improve the student compared with corpus-only training?
- Is teacher-only distillation enough, or does the student still need real corpus labels?
- Which distillation temperature gives the best validation perplexity?
- What is the quality-efficiency trade-off between GPT-2 teacher and the smaller students?

## Success Criteria

- all three student models train successfully with the same architecture
- the temperature sweep selects a reasonable temperature based on validation perplexity
- the comparison shows whether `Teacher + corpus` improves over `Corpus only` and `Teacher only`
- the students are smaller than the GPT-2 teacher by parameter count and disk size
- the students are faster than the teacher during generation
- validation/test perplexity, training loss, and generated samples give a clear picture of model quality

## Why Temperature Matters

In distillation, the teacher produces logits for every possible next token. Applying softmax directly can make the distribution very sharp, meaning most probability goes to only one or two tokens. Temperature softens this distribution:

```text
higher temperature -> softer probability distribution
lower temperature  -> sharper probability distribution
```

A softer distribution can reveal the teacher's uncertainty. For example, the teacher may strongly prefer one token but still assign meaningful probability to related tokens. This extra information is called soft-target information, and it can help the student learn more than it would from only the single correct token.

However, if the temperature is too high, the distribution can become too flat and less useful. That is why this project uses a temperature sweep: we want to find a value that is soft enough to transfer useful teacher knowledge, but not so soft that the teacher signal becomes weak.

## How Loss Is Calculated

All three student models use the same input text blocks, but they calculate training loss differently.

### 1. Teacher + Corpus

This student uses both the real corpus labels and the teacher's soft targets:

```text
total_loss = alpha * corpus_loss + (1 - alpha) * teacher_loss
```

- `corpus_loss`: cross-entropy between the student prediction and the true next token from the dataset.
- `teacher_loss`: KL divergence measuring how far the student's probability distribution is from the teacher's, with the teacher as the target distribution.
- `alpha`: controls how much weight is given to the real corpus labels versus the teacher signal. This project uses `alpha = 0.5`.

### 2. Corpus Only

This student ignores the teacher and learns only from the dataset:

```text
total_loss = corpus_loss
```

This is the normal language-model training baseline.

### 3. Teacher Only

This student ignores the hard labels from the corpus and learns only from the teacher:

```text
total_loss = teacher_loss
```

This tests whether imitating the teacher distribution alone is enough.

### Loss Details

`corpus_loss` uses cross-entropy because the dataset provides one correct next token. `teacher_loss` uses KL divergence because the teacher provides a full probability distribution over the vocabulary.

For `teacher_loss`, the logits are divided by the temperature before softmax:

```text
student distribution = softmax(student logits / T)
teacher distribution = softmax(teacher logits / T)
```

Following the Hinton distillation paper, the KL-divergence loss is multiplied by `T²`:

```text
teacher_loss = KL(teacher distribution || student distribution) * T²
```

The order matters. The teacher is the target distribution and the student is the one being fitted, so this is the forward KL, written `KL(teacher || student)`. In code it is `F.kl_div(student_log_probs, teacher_probs)`, because PyTorch's `kl_div` takes the log-probabilities of the fitted distribution first and the target distribution second.

This matters because increasing temperature makes the soft-target gradients smaller. Multiplying by `T²` keeps the teacher-loss scale more comparable when testing different temperatures.

## Temperature Sweep Design

The sweep trains one throwaway student per temperature and ranks the temperatures by validation perplexity. Two design choices are worth explaining.

**The sweep uses `teacher_only`, not `Teacher + corpus`.** Temperature enters the loss only through `teacher_loss`. With the combined objective and `alpha = 0.5`, half of every gradient comes from `corpus_loss`, and that half is identical at every temperature. It contributes signal that has nothing to do with the variable being swept, which flattens the differences between temperatures and can leave the sweep curve indistinguishable from noise. Training on the teacher signal alone makes the sweep a direct measurement of the quantity we care about: how much usable knowledge each temperature transfers from the teacher.

**Temperatures are still ranked by validation perplexity**, which is measured against the real next tokens in the validation split. So the sweep answers a well-posed question: at which temperature does imitating the teacher transfer best to real next-token prediction?

The trade-off is that the temperature is chosen under the teacher-only objective and then reused by the final `Teacher + corpus` student, whose loss is different. The winning temperature is therefore not guaranteed to be optimal for the blended loss. We accept this because a clean, measurable temperature signal is more useful than a damped one, and because sweeping temperature and `alpha` jointly would multiply the training cost. The absolute perplexities reported in the sweep table are also worse than the final students', since sweep runs never see the corpus labels.

One reading note: the sweep table's training-loss column is `KL * T²`, which grows with temperature by construction. It cannot be compared across rows, and the validation-perplexity columns are the ones to read.

## Early Stopping

The project uses early stopping inside `train_model`. After each epoch, the student is evaluated on the validation set:

```text
validation_ppl = evaluate_perplexity(model, val_loader)
```

If validation perplexity improves, training continues. If validation perplexity does not improve for `early_stopping_patience` epochs in a row, training stops:

```text
stop when validation perplexity stops improving
```

This project uses `early_stopping_patience = 2` and a budget of at most 15 epochs, so a student stops as soon as two consecutive epochs fail to beat its best validation perplexity.

This helps prevent overfitting. If training loss keeps decreasing but validation perplexity stops improving, the student may be memorizing the training data instead of learning patterns that generalize.

## Dataset

The project uses **`wikitext-2-raw-v1`** from Hugging Face (`Salesforce/wikitext`) for every experiment: the temperature sweep, the final training of all three students, and all evaluation. This is the only dataset used. WikiText-103 was considered but rejected: at roughly 100M tokens it is about 40x larger, and the run already trains 11 models (8 sweep students plus 3 final students), each of which needs a GPT-2 teacher forward pass on every batch.

The raw split sizes are approximately 2.4M GPT-2 tokens for train, 250K for validation, and 280K for test. Data preparation concatenates all tokenized text into one stream and cuts it into non-overlapping blocks of `block_size` tokens, which gives roughly 2,300 training blocks of 1024 tokens. Tokens left over at the end of each internal batch do not fill a whole block and are dropped, which discards on the order of 1% of the corpus.

Because the corpus is small relative to a from-scratch Transformer, overfitting is a genuine risk, and it is one of the things the validation-perplexity curve is meant to expose.

## Configuration

The values below are the ones in `CFG` at the top of the notebook.

| Setting | Value |
|---|---|
| Dataset | `wikitext-2-raw-v1` |
| Tokenizer | GPT-2 BPE, 50,257 tokens (shared by teacher and students) |
| `block_size` | 1024 tokens (GPT-2's full context length) |
| `batch_size` | 16 blocks, so 16,384 tokens per optimizer step |
| Teacher | `gpt2` (GPT-2 small), 124.4M parameters |
| Student | 6 layers, hidden size 384, 6 heads, 30.3M parameters |
| Optimizer | AdamW, `lr = 5e-4` |
| Epochs | up to 15, early stopping with patience 2 |
| Sweep epochs | up to 10 per temperature |
| `alpha` | 0.5 (equal weight on corpus loss and teacher loss) |
| Temperatures swept | 1, 2, 4, 7, 10, 15, 20, 30 |
| Seed | 42 |

Both models use the same 1024-token context, so teacher and student perplexity are measured on identical inputs.

On parameter count: the student is 4.1x smaller than the teacher overall (30.3M vs 124.4M), but that number is held back by the embedding tables, which cannot shrink because the tokenizer is shared. The student's token and position embeddings alone are 19.7M of its 30.3M parameters. Comparing only the Transformer stack, the student is **8x smaller** (10.6M vs 85.1M).

## Evaluation Metrics

The project reports:

- **Validation perplexity**: tracked every epoch, used for early stopping and for choosing the temperature.
- **Test perplexity**: final language-modeling quality, measured once at the end.
- **Tokens/sec**: generation speed, measured over repeated fixed-length generations.
- **Parameter count**: model size.
- **Disk size**: saved checkpoint size.
- **Generated samples**: the same prompts given to the teacher and all three students with greedy decoding.

Perplexity is computed on non-overlapping 1024-token blocks, so the first tokens of each block are predicted with no preceding context. This is a standard way to measure it, but it means these numbers are not directly comparable to published GPT-2 results that use a sliding window.

Section 10 of the notebook plots four panels: training loss per epoch, validation perplexity per epoch, final test perplexity, and inference speed. The validation-perplexity panel is the one to read when comparing the three methods, because the three training losses are different quantities and cannot be compared with each other.

## How The Three Students Are Compared

The three student models are compared under the same setup:

- same student architecture
- same initial weights (all three students are built from the same random seed)
- same training dataset, visited in the same shuffled order
- same tokenizer
- same optimizer and learning rate
- same selected distillation temperature
- same epoch budget, with the same early-stopping rule

The random seed is reset at the start of every training run, so the three students also see
identical dropout masks and identical batch ordering. This matters because we train each student
only once: if the data order differed between runs, part of any perplexity gap we measure would
come from the shuffle rather than from the loss function.

Note that the students do **not** all run for the same number of epochs. Each one trains until its
own validation perplexity stops improving, so every method is given the chance to reach its own best
result instead of being cut off at a fixed epoch. The report lists the number of epochs each student
actually completed.

The only difference is the training loss:

- `Teacher + corpus`: corpus loss plus teacher loss
- `Corpus only`: corpus loss only
- `Teacher only`: teacher loss only

The comparison focuses on these questions:

1. **Does distillation help?**
   - Compare `Teacher + corpus` with `Corpus only`.
   - If `Teacher + corpus` has lower validation/test perplexity, the teacher signal helped.

2. **Is teacher-only training enough?**
   - Compare `Teacher only` with `Teacher + corpus`.
   - If `Teacher only` performs worse, then the real corpus labels are still important.

3. **How much quality is lost compared with the teacher?**
   - Compare each student with GPT-2 teacher on test perplexity and generated samples.

4. **How much efficiency is gained?**
   - Compare parameter count, disk size, and tokens/sec between the teacher and students.

5. **Is there possible overfitting?**
   - Compare final training loss with validation/test perplexity.
   - Low training loss but high validation/test perplexity suggests overfitting.

