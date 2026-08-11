# Knowledge Distillation of GPT-2 Small

This is our ECE1508 course project on compressing GPT-2 with knowledge distillation. We use the pretrained 124.4M parameter GPT-2 small model as a teacher and train a 30.3M parameter GPT-2 style student on WikiText-2.

The main question is simple: how much language modeling quality can a smaller model keep, and what do we gain in model size and generation speed?

The full experiment is in [`distillation.ipynb`](distillation.ipynb). It includes data preparation, training, evaluation, generated examples, and plots.

## What we compare

We train the same student architecture in three ways:

1. **Teacher + corpus** uses both the true next token and the teacher's probability distribution.
2. **Corpus only** uses only the true next token. This is our normal language modeling baseline.
3. **Teacher only** learns only from the teacher's probability distribution.

The students use the same initial weights, data, tokenizer, optimizer, and training settings. The loss is the main difference between the runs.

## Project design

### Models

| Model | Layers | Hidden size | Heads | Parameters |
|---|---:|---:|---:|---:|
| GPT-2 small teacher | 12 | 768 | 12 | 124.4M |
| Student | 6 | 384 | 6 | 30.3M |

The teacher is frozen during training, so only the student is updated. Both models use the GPT-2 tokenizer, its 50,257 token vocabulary, and a context length of 1024 tokens.

### Data and training

We use `Salesforce/wikitext` with the `wikitext-2-raw-v1` configuration. The text is tokenized, joined into one stream, and split into non-overlapping blocks of 1024 tokens.

The final runs use:

- AdamW with a learning rate of `5e-4`
- batch size 16
- up to 50 epochs
- early stopping with patience 2
- random seed 42

After training, the notebook restores the checkpoint with the lowest validation perplexity before running the final test.

### Losses

The corpus loss is cross entropy against the true next token. The teacher loss uses forward KL divergence to match the student's distribution to the teacher's distribution.

```text
corpus only:
    total_loss = corpus_loss

teacher only:
    total_loss = teacher_loss

teacher + corpus:
    total_loss = alpha * corpus_loss + (1 - alpha) * teacher_loss
```

We use `alpha = 0.5`. The teacher loss also applies temperature scaling and the usual `T²` correction.

### Temperature sweep

Before the final runs, we test `T = 1, 2, 4, 7, 10, 15`. Each temperature gets a temporary student trained for five epochs with the teacher-only loss. We compare their validation perplexities and use the best temperature for final training.

These temporary models are only used to rank temperatures, so there is no need to train every one to completion.

## Setup

The final experiment was run with Python 3.12 on an AMD Radeon PRO W7900 with 48 GB of memory.

Create a virtual environment from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the Python packages used by the notebook:

```powershell
python -m pip install torch transformers datasets numpy pandas matplotlib tqdm notebook
```

Make sure the PyTorch build matches your hardware. The default package may not provide GPU support for every system. For the W7900 run, we used an AMD ROCm build of PyTorch.

The first run also needs internet access to download GPT-2 and WikiText-2 from Hugging Face.

## Running the project

Start the notebook with:

```powershell
jupyter notebook distillation.ipynb
```

Select the virtual environment as the kernel and run the cells from top to bottom. The notebook will:

1. download and prepare the dataset;
2. load the GPT-2 teacher;
3. run the temperature sweep;
4. train the three students;
5. evaluate perplexity, speed, size, and generated text;
6. save the plots and student checkpoints.

The full run is expensive and is intended for a high memory GPU. If memory is limited, reduce `batch_size` in `CFG`.

## Results

The [`results/`](results/) folder contains the project plots and notebook export.

- [`temperature_sweep.png`](results/temperature_sweep.png) shows validation perplexity for the tested temperatures.
- [`comparison.png`](results/comparison.png) contains the training curves and final model comparison.
- [`distillation.pdf`](results/distillation.pdf) is a PDF export of the executed Jupyter notebook. It contains the saved output from the final run, including the temperature sweep, training histories, final metrics, and generated examples.

The final written report is available at [`doc/final_report.pdf`](doc/final_report.pdf).

## Repository layout

```text
distillation.ipynb    Main experiment notebook
README.md             Setup and project overview
doc/                  Proposal and final report
reference/            Reference papers and PDFs
results/              Output plots and executed notebook PDF
```
