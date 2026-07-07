# GPT-2 Knowledge Distillation Project

This project studies knowledge distillation for language models. A pretrained GPT-2 small model is used as the teacher, and smaller GPT-2-style student models are trained and compared.

## Project Goal

The goal is to compress GPT-2 into a smaller student model while keeping reasonable language-modeling quality. The project compares three student training methods:

1. **Teacher + corpus**: learns from both the real text corpus and the teacher's soft probability distribution.
2. **Corpus only**: learns only from the real next-token labels in the dataset.
3. **Teacher only**: learns only from the teacher's soft probability distribution.

The project also runs a **temperature sweep** before final training. Temperature controls how soft the teacher's output distribution is during distillation. We test several values and choose the one with the best validation perplexity before training the final students. This helps us learn how sensitive distillation is to temperature and whether softer teacher probabilities improve student learning.

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
- `teacher_loss`: KL divergence between the student probability distribution and the teacher probability distribution.
- `alpha`: controls how much weight is given to the real corpus labels versus the teacher signal.

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
teacher_loss = KL(student distribution, teacher distribution) * T²
```

This matters because increasing temperature makes the soft-target gradients smaller. Multiplying by `T²` keeps the teacher-loss scale more comparable when testing different temperatures.

## Early Stopping

The project uses early stopping inside `train_model`. After each epoch, the student is evaluated on the validation set:

```text
validation_ppl = evaluate_perplexity(model, val_loader)
```

If validation perplexity improves, training continues. If validation perplexity does not improve for `early_stopping_patience` epochs, training stops:

```text
stop when validation perplexity stops improving
```

This helps prevent overfitting. If training loss keeps decreasing but validation perplexity stops improving, the student may be memorizing the training data instead of learning patterns that generalize.

## Dataset

The notebook uses WikiText-103 from Hugging Face for final training:

```python
load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
```

WikiText-103(~100M tokens) is larger than WikiText-2(~2M tokens), so it gives the student more training text. This should help the student learn more general language patterns and reduce the chance of memorizing a small training set.

For quick debugging, WikiText-2 can still be used by changing:

```python
CFG["dataset_config"] = "wikitext-2-raw-v1"
```

## Evaluation Metrics

The project reports:

- **Validation perplexity**: used to check model quality during analysis.
- **Test perplexity**: final language-modeling quality.
- **Tokens/sec**: generation speed.
- **Parameter count**: model size.
- **Disk size**: saved checkpoint size.
- **Generated samples**: qualitative comparison of model outputs.

## How The Three Students Are Compared

The three student models are compared under the same setup:

- same student architecture
- same training dataset
- same tokenizer
- same number of epochs
- same optimizer and learning rate
- same selected distillation temperature

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

