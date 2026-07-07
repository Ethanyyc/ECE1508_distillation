# Project Proposal: I wi for Language Models

## Introduction

This project will study knowledge distillation for language models by compressing a teacher model into a smaller student model. The goal is to reduce inference cost while preserving generation quality and task performance. This matters in real applications such as chatbots and writing assistants, where low latency and small model size are important for mobile or edge deployment and lower cloud cost. By comparing a teacher-student setup, we can measure how much capability can be transferred into a smaller model and what trade-offs appear in speed, accuracy, and size.

## Course Topics Covered

This project covers **Chapter 1, Transformer-based LMs and LLMs / Transformer LMs**, and **Chapter 2, Fundamentals of Generative Modeling / Generative Modeling**. The Transformer LM topic is covered because both teacher and student will be transformer-based language models, and we will study how self-attention supports sequence modeling. The generative modeling topic is covered because a language model learns a text distribution and generates new tokens one by one. Distillation fits this setting because the student learns from both ground-truth text and the teacher's soft output distribution, linking transformer architecture with generative learning.

## Approach

We will begin by choosing a teacher model and a smaller student model. For this project, we plan to use GPT-2 small as the teacher model, since it is around 0.1B parameters, large enough to demonstrate distillation but still feasible to run on course-level hardware. The student will be a smaller transformer, likely in the 25M to 50M parameter range, with fewer layers and a smaller hidden dimension. We will use a text dataset that is manageable for course-scale experimentation, such as a subset of WikiText-2 or WikiText-103. The text will first be tokenized into model input tokens using the same tokenizer for both teacher and student. During training, each batch of tokenized text will be passed through the teacher model to obtain soft predictions. The student will then learn from two signals: the correct next word from the dataset and the teacher's full probability distribution over possible next words. After computing the loss, we will use backpropagation and an optimizer step to update only the student model parameters, while keeping the teacher fixed in evaluation mode. This helps the student learn both the real text and the teacher's behavior.

For implementation, we will use Python, PyTorch, Hugging Face Transformers, and the Datasets library. Evaluation will include:

- **Perplexity:** lower is better for language models, so this will measure next-token prediction quality on a held-out test set.
- **Generated text quality:** we will give both models the same prompt and compare fluency, coherence, and repetition in the outputs.
- **Inference speed:** we will measure how long each model takes to generate the same number of tokens.
- **Model size and parameter count:** we will report total parameters and memory footprint to show how much smaller the student is.
- **Task-specific accuracy:** if we add a downstream task such as classification or question answering, we will compare accuracy there too.

## Key Outputs of the Project

The main output of this project will be a trained student language model that has been distilled from GPT-2 small. We will also produce a side-by-side comparison of teacher and student performance, including perplexity, generated text quality, inference speed, and model size. In addition, we will summarize the training process, report the distillation results, and discuss how much performance is kept after compression. These outputs will show both the practical effect of distillation and the trade-off between efficiency and quality.
