# References

## Knowledge Distillation

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. **Distilling the Knowledge in a Neural Network**. arXiv:1503.02531, 2015. <https://arxiv.org/abs/1503.02531>
  - Classic paper introducing knowledge distillation using soft targets, temperature scaling, and training a smaller student model from a larger teacher or ensemble.

- Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. arXiv:1910.01108, 2019. <https://arxiv.org/pdf/1910.01108>
  - Transformer distillation paper that combines language modeling loss, teacher soft-target distillation loss, and hidden-state alignment. Useful for justifying a combined hard-label and soft-label training objective.
