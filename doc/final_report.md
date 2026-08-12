---
title: "Knowledge Distillation of GPT-2 Small: Compressing a 124M Teacher into a 30M Student"
author:
  - |
    **Yicheng Yao**\
    `yicheng.yao@mail.utoronto.ca`
  - |
    **Jarvis Wang**\
    `jarvis.wang@mail.utoronto.ca`
  - |
    **Jiangchuan Yu**\
    `jiangchuan.yu@mail.utoronto.ca`
date: "ECE1508 — Deep Generative Models, Summer 2026"
geometry: "margin=0.75in"
fontsize: 10pt
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
  - \AtBeginEnvironment{tabular}{\small}
  - \setlength{\intextsep}{6pt plus 2pt minus 2pt}
  - \setlength{\textfloatsep}{6pt plus 2pt minus 2pt}
  - \let\origfigure\figure
  - \let\endorigfigure\endfigure
  - \renewenvironment{figure}[1][]{\origfigure[H]}{\endorigfigure}
---

\begin{abstract}
Large language models are powerful but expensive to run, so we wanted to see how much of GPT-2's ability survives when it is shrunk to a quarter of its size. We use a pretrained GPT-2 small as the "teacher" (124.4M parameters) and train a much smaller "student" (30.3M parameters) from scratch on WikiText-2. The architecture, data, and training loop are held fixed, and we compare three ways of training the student that differ only in the loss: learning from the real text (corpus only), from the teacher's soft predictions (teacher only), or from both. We also sweep the distillation temperature. Using the teacher helps a lot, roughly halving the test perplexity compared with training on the text alone. The teacher-only student came out best, at a test perplexity of 90.7, ahead of the combined loss (115.6) and well ahead of corpus-only (227.2). The student ends up about 4.1$\times$ smaller than the teacher, a quarter of the disk size, and roughly 1.8$\times$ faster to generate, at the cost of higher perplexity.
\end{abstract}

# Attestation of Teamwork

All three members contributed to the project. **Yicheng Yao** wrote the distillation loss functions (the three objectives, the temperature scaling, and the $T^2$ correction), the training loop with early stopping and best-weight restoration, and the temperature sweep. **Jarvis Wang** built the data pipeline (tokenizing and blocking WikiText-2), set up the teacher and student models, and wrote the evaluation code for perplexity, text generation, and speed. **Jiangchuan Yu** ran all the experiments, made the figures and tables, and led the writing of this report and the slides. 

# Introduction

Modern language models work well but are heavy to deploy. They are slow, use a lot of memory, and cost money to serve, which is a real problem when a model has to run on a phone, on cheap hardware, or under a tight latency budget. **Knowledge distillation** [1] is a well-known way to make a model smaller without starting over: we train a small *student* model to copy a larger *teacher* model.

What makes distillation work is that the teacher gives more than just the right answer. For every next word it produces a full probability distribution over the vocabulary, and even when the top choice is obvious the smaller probabilities it assigns to the other words are informative. Hinton et al. call this extra signal *dark knowledge*, and a student that learns from it can pick up how the teacher generalizes rather than just which single word is correct.

We apply this idea to GPT-2 small on WikiText-2. The aim is to measure, in a controlled setup, how much of the teacher's language ability transfers into a much smaller student and what we get back in speed and size.

# Preliminaries and Problem Formulation

**Setup.** Let $P_T(\cdot\mid x_{<t})$ and $P_S(\cdot\mid x_{<t})$ be the next-token distributions of the teacher and student given the words so far. The teacher is a frozen, pretrained GPT-2 small; the student is a smaller GPT-2-style model trained from scratch. They share the same GPT-2 tokenizer (50,257 tokens) and the same 1024-token context, because to compare or copy distributions the two models must be talking about the same tokens at the same positions.

**Goal.** Train smaller and faster student models that still perform well on unseen validation and test text. The following five questions help us compare model quality and efficiency:

1. Does using the teacher beat plain training on the text?
2. Is the teacher signal *alone* enough, or do we still need the real labels?
3. Which temperature transfers the teacher's knowledge best?
4. How far behind the teacher does the student end up?
5. How much smaller and faster do we get?

**Perplexity.** We measure quality with perplexity, $\mathrm{PPL}=\exp\!\big(\tfrac{1}{N}\sum_t -\log P(x_t\mid x_{<t})\big)$, which is just the exponential of the average per-token cross-entropy on the real next words. Lower is better.

# Design

**Models.** The teacher is `gpt2` (GPT-2 small): 12 layers, hidden size 768, 12 heads, 124.4M parameters. The student keeps the same style but halves both the depth and the width: 6 layers, hidden size 384, 6 heads, 30.3M parameters.

**Where the parameters come from.** It helps to split a GPT-2 model into two parts. The **embedding tables** hold one vector of length $d$ (the hidden size) for every token and every position, so they cost about $(V + L_{\text{ctx}})\times d$ parameters, where $V=50{,}257$ and $L_{\text{ctx}}=1024$. Following the parameter-count approximation from Kaplan et al. [3], each standard **Transformer block** has about $12d^2$ parameters, so a stack of $n$ blocks has about $12\,n\,d^2$. This approximation covers the attention and feed-forward weights while leaving out small terms such as biases. Plugging in the numbers:

- Teacher: embeddings $(50{,}257+1024)\times 768 \approx 39.4\text{M}$; stack $12\times 12\times 768^2 \approx 85.1\text{M}$; total $\approx 124.4\text{M}$.
- Student: embeddings $(50{,}257+1024)\times 384 \approx 19.7\text{M}$; stack $12\times 6\times 384^2 \approx 10.6\text{M}$; total $\approx 30.3\text{M}$.

The key point is that the two parts shrink at different rates (Appendix A). The embeddings only halve, because their row counts (50,257 tokens, 1024 positions) are fixed by the shared tokenizer and context, so only the width $d$ changes. The Transformer stack shrinks by $8\times$: halving the width quarters each block ($d^2$) and halving the depth removes half the blocks, giving $4\times 2 = 8$. So the headline number is only $4.1\times$ even though the part that does the real work is $8\times$ smaller — which will matter for speed.

**Three objectives, one pipeline.** We train three students that are identical in every way (architecture, starting weights, data order) except for the loss. Writing $\mathrm{CE}$ for the cross-entropy against the true next word, the distillation term is
$$
\mathcal{L}_{\text{KD}} = T^2\cdot \mathrm{KL}\big(\,\mathrm{softmax}(z_T/T)\ \|\ \mathrm{softmax}(z_S/T)\,\big),
$$
where $\mathcal{L}_{\text{KD}}$ is the KL-divergence-based knowledge-distillation loss, $z_T$ and $z_S$ are the teacher and student logits, and $T$ is the temperature. The three objectives are:
$$
\text{Corpus only: } \mathcal{L}=\mathrm{CE};\quad
\text{Teacher only: } \mathcal{L}=\mathcal{L}_{\text{KD}};\quad
\text{Teacher + corpus: } \mathcal{L}=\alpha\,\mathrm{CE} + (1-\alpha)\,\mathcal{L}_{\text{KD}},
$$
with $\alpha=0.5$. The pipeline is the same every time: run the batch through the frozen teacher and the student, compute one loss, and update only the student.

**The $T^2$ factor.** For any positive temperature $T$, we define the distillation loss as
$$
\mathcal{L}_{\mathrm{KD}}(T)
=T^2\,
\mathrm{KL}\left(
\mathrm{softmax}(z_T/T)
\,\|\, 
\mathrm{softmax}(z_S/T)
\right).
$$
Writing $p(T)=\mathrm{softmax}(z_S/T)$, $q(T)=\mathrm{softmax}(z_T/T)$, and $\mathcal{L}_{\mathrm{KL}}$ for the same loss without the factor, the student gradient is
$$
\frac{\partial\mathcal{L}_{\mathrm{KL}}}{\partial z_{S,i}}
=\underbrace{\frac{1}{T}}_{\text{chain rule}}\underbrace{\big(p_i(T)-q_i(T)\big)}_{O(1/T)}
=O\!\left(\frac{1}{T^{2}}\right),
\qquad\text{so}\qquad
\frac{\partial\mathcal{L}_{\mathrm{KD}}}{\partial z_{S,i}}
=T\big(p_i(T)-q_i(T)\big)=O(1).
$$
One $1/T$ comes from the chain rule, since the softmax input is $z_{S,i}/T$, and a second comes from the bracket, because a higher $T$ flattens both distributions and shrinks their difference. The $T^2$ factor cancels both **in the large-$T$ regime**, keeping the teacher signal at a comparable strength for $T \gtrsim 1$. This matters because the corpus loss uses unscaled logits and does not shrink with $T$: without the correction, the sweep would measure gradient size rather than the information in the softened targets. At $T=1$ the factor equals one, so our final runs are unchanged.

The cancellation is asymptotic, however, and is not guaranteed below $T=1$. The step $p_i(T)-q_i(T)=O(1/T)$ assumes $T$ is large enough that both distributions are near-uniform; as $T\to 0$ they instead sharpen toward one-hot and their difference saturates. Probing a cross-entropy-trained student confirms the correction is imperfect there: with the $T^2$ factor applied, the KD gradient norm measured 2.784 at $T=0.5$ against 0.739 at $T=1$, so the teacher signal arrives roughly $3.8\times$ stronger. A mismatch of that size would confound temperature with effective learning rate, so we treat it as a threat to validity and test it directly with a gradient-matched control run in the $\alpha$ and temperature grid.

# Methodology

**Data.** We use `wikitext-2-raw-v1` from Hugging Face (about 2.4M GPT-2 tokens) for everything. We tokenize the text, join it into one long stream, and split it into non-overlapping 1024-token blocks. GPT-2 was not trained or fine-tuned on WikiText-2, so the teacher is tested zero-shot. The students are trained on WikiText-2, which gives them an advantage. A teacher trained on the same data would likely have lower perplexity, so the actual gap between the teacher and students may be larger.

**Training.** Each student is trained with AdamW (`lr = 5e-4`), batch size 16 (16,384 tokens per step), for up to 50 epochs with early stopping (patience 2). After each epoch, we check validation perplexity and save the model weights when it improves. For final testing, we use the weights from the epoch with the lowest validation perplexity, which is the best validation checkpoint. To make the three-way comparison fair, the training function re-seeds itself on entry (seed 42), so all three students see the same starting weights, the same batch order, leaving the loss as the only difference.

**Temperature sweep.** Temperature is important because it changes what the student learns from the teacher. A low temperature mainly focuses on the teacher's most likely token, while a higher temperature shows more information about other possible tokens. However, if the temperature is too high, the distribution becomes too flat and the teacher signal becomes less useful. The best value depends on the teacher, student size, and dataset, so it cannot be chosen confidently without testing. This experiment shows how sensitive our distillation method is to temperature and helps us choose the strongest teacher signal for the final comparison. We test $T\in\{1,2,4,7,10,15\}$. Each sweep run trains a throwaway student for 5 epochs with the **teacher-only** loss, so temperature is the only factor changing the gradient. We do not fully train every sweep model because the goal is only to rank the temperatures under the same short training budget. We choose the value with the lowest validation perplexity and use it for the longer final training runs. A limitation is that the ranking could change if every sweep model were trained for more epochs.

**Implementation.** PyTorch with Hugging Face `transformers` (`GPT2LMHeadModel`, `GPT2TokenizerFast`) and `datasets`. The KL term uses `F.kl_div` with `reduction = "batchmean"`, which averages over tokens so it sits on the same scale as the mean cross-entropy. GPT-2 uses the output at each position to predict the following token. Therefore, we shift the logits and labels by one position: the prediction at position $t$ is compared with the token at position $t+1$. Without this shift, the model would be trained to predict the token it already received instead of the next token. Runs used an AMD Radeon PRO W7900 (48 GB, ROCm).

# Numerical Experiments

The settings shared by every run are listed in Appendix B.

**Picking the temperature.** Figure 1 shows the sweep. Under the teacher-only loss, validation perplexity is lowest at $T=1$ and climbs steeply as the temperature goes up, from about 258 at $T=1$ to about 2191 at $T=15$. The curve is basically flat and bad past $T=10$. The reading is that softening the teacher hurts here rather than helps, so we set $T=1$ for the final students. (The absolute numbers are high because each sweep run is short, we only use them to *rank* the temperatures, not as final scores.) We say more about *why* $T=1$ wins in the Discussion.

\begin{figure}[t]
\vspace{-6pt}
\centering
\includegraphics[width=0.5075\linewidth]{../results/temperature_sweep.png}
\vspace{-8pt}
\caption{Temperature sweep result}
\end{figure}

**The main comparison.** Table 1 summarizes the final results, while Figure 2 shows the training curves, test perplexity, and generation speed. The three students use the same architecture, initial weights, data, and training settings. Their training objective is the main difference, so the results show how each loss affects the same student model.

The validation perplexity panel in the upper right of Figure 2 gives the clearest comparison during training. All three students improve quickly during the first few epochs, but their results later separate. Corpus only reaches its best validation perplexity of 214.5 at epoch 13 and then begins to get worse. Teacher + corpus continues improving until epoch 22 and reaches 111.2. Teacher only improves until epoch 39 and reaches 89.4, which is the best student result. The circles show the checkpoint selected for final testing. These curves suggest that teacher guidance helps the student train longer before overfitting. The training loss panel in the upper left should not be compared directly because Corpus only uses cross entropy, Teacher only uses KL divergence, and Teacher + corpus uses a weighted mixture of both losses.

The test perplexity panel in the lower left shows the final quality of each model. GPT 2 has the lowest test perplexity at 29.4. Among the students, Teacher only performs best at 90.7, followed by Teacher + corpus at 115.6 and Corpus only at 227.2. Compared with Corpus only, Teacher + corpus lowers test perplexity by about 49 percent, while Teacher only lowers it by about 60 percent. This shows that the teacher distribution gives the student useful information that is not available from the true next token alone.

The generation speed panel in the lower right shows that all three students run at about 292 to 295 tokens per second, compared with 164.6 tokens per second for GPT 2. Each student has 30.3M parameters and uses about 119.1 MB on disk, while the teacher has 124.4M parameters and uses about 474.7 MB. The students have almost the same speed and size because they share the same architecture. Their training objective changes model quality but does not change inference cost.

\begin{table}[htbp]
\centering
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
Model & Params (M) & Disk (MB) & Val PPL & Test PPL & Tok/s & Best ep. \\
\midrule
Teacher (GPT-2) & 124.4 & 474.7 & 30.6 & 29.4 & 164.6 & --- \\
Student: Teacher only & 30.3 & 119.1 & \textbf{89.4} & \textbf{90.7} & 292.3 & 39 \\
Student: Teacher + corpus & 30.3 & 119.1 & 111.2 & 115.6 & 295.1 & 22 \\
Student: Corpus only & 30.3 & 119.1 & 214.5 & 227.2 & 294.3 & 13 \\
\bottomrule
\end{tabular}
\caption{Final results}
\end{table}

\begin{figure}[t]
\centering
\vspace{-8pt}
\makebox[\linewidth][c]{\includegraphics[width=1.10\linewidth]{../results/comparison.png}}
\vspace{-8pt}
\caption{Full training results}
\end{figure}

**What the text looks like.** With greedy decoding, the teacher writes fluent sentences while all three 30M students fall into repetition, expected at this size. Still, the teacher-guided students are noticeably less broken than the corpus-only one, which collapses into tight loops (for example, repeating "city" over and over). A couple of transcripts are in Appendix C.

## The $\alpha$ and Temperature Grid

The experiments above fix $\alpha=0.5$ and pick $T$ from a five-epoch sweep, which leaves two questions open: what $\alpha$ actually does, and what happens when the teacher distribution is *sharpened* ($T<1$) rather than softened. We therefore trained eight more students, each to early stopping under the identical recipe (Appendix B), on a single machine so that every point is mutually comparable.

Because $\alpha$ weights the corpus term in $\mathcal{L}=\alpha\,\mathrm{CE}+(1-\alpha)\,\mathcal{L}_{\text{KD}}$, the three students of Table 1 already sit at $\alpha=1$ (corpus only), $\alpha=0.5$ (combined) and $\alpha=0$ (teacher only). The grid fills in $\alpha\in\{0.25,0.75\}$ at $T=1$ to complete a five-point curve, and adds $T\in\{0.5,2\}$ at $\alpha=0.5$ under **full** training rather than the five-epoch budget of Figure 1. The three re-run baselines land close to Table 1 despite the different machine (corpus-only 217.5 vs 214.5, combined 108.2 vs 111.2, teacher-only 87.1 vs 89.4), so the two sets of numbers tell the same story.

\begin{table}[htbp]
\centering
\scriptsize
\begin{tabular}{rrcrrrr}
\toprule
$\alpha$ & $T$ & Grad-matched & Best ep. & Best val PPL & Test PPL & Val PPL @15 ep \\
\midrule
0.00 (teacher only) & 1 & no & 50$^{*}$ & 87.10 & 88.06 & 113.92 \\
\textbf{0.25} & 1 & no & 43 & \textbf{83.22} & \textbf{85.59} & \textbf{104.84} \\
0.50 (combined) & 1 & no & 22 & 108.21 & 112.60 & 114.93 \\
0.75 & 1 & no & 17 & 145.62 & 152.44 & 145.66 \\
1.00 (corpus only) & 1 & no & 14 & 217.50 & 226.90 & 217.50 \\
\midrule
0.50 & 0.5 & no & 19 & 148.26 & 155.96 & 153.75 \\
0.50 & 2 & no & 21 & 107.43 & 110.90 & 119.59 \\
0.50 & 0.5 & \textbf{yes} & 21 & 147.89 & 153.95 & 158.85 \\
\bottomrule
\end{tabular}
\caption{The $\alpha$ and temperature grid. $^{*}$reached the 50-epoch cap while still improving.}
\end{table}

**Effect of $\alpha$: the optimum is interior.** Test perplexity falls from 226.90 at $\alpha=1$ to 152.44 at $\alpha=0.75$ and 112.60 at $\alpha=0.5$, reaches its minimum of **85.59 at $\alpha=0.25$**, then rises slightly to 88.06 at $\alpha=0$. Pure distillation is therefore not quite optimal: keeping a *small* amount of hard-label grounding beats discarding the corpus signal entirely. The curve is strongly asymmetric. Moving from the optimum toward the teacher costs only 2.5 perplexity ($\alpha=0.25\to0$), whereas moving the same distance toward the corpus costs 67 ($\alpha=0.25\to0.75$). The practical reading is that the teacher signal should dominate the objective, with the corpus acting as a light anchor rather than an equal partner.

The stopping epochs make the mechanism visible. As $\alpha$ increases, the best epoch falls monotonically -- 50, 43, 22, 17, 14 -- so the more hard-label pressure the objective carries, the sooner the student overfits WikiText-2. This is direct support across five points for reading soft targets as a regularizer.

\begin{figure}[t]
\centering
\vspace{-6pt}
\makebox[\linewidth][c]{\includegraphics[width=1.10\linewidth]{../results/alpha_sweep.png}}
\vspace{-8pt}
\caption{Effect of $\alpha$ at $T=1$. Left: perplexity against $\alpha$, with the fixed 15-epoch budget overlaid. Middle: validation curves. Right: stopping epoch, which falls monotonically as $\alpha$ rises.}
\end{figure}

**How to read the $\alpha$ curve honestly.** Because early stopping fires at very different epochs, the raw curve partly reflects *how long each objective trains* rather than its intrinsic quality. We therefore also record the best validation perplexity within a fixed 15-epoch budget. The ranking is unchanged -- $\alpha=0.25$ still leads at 104.84, ahead of $\alpha=0$ at 113.92 and $\alpha=0.5$ at 114.93 -- so the advantage is not merely an artifact of training longer. One limitation remains: $\alpha=0$ was still improving when it hit the 50-epoch cap, so with a larger budget the two best points could converge or swap. With a single seed we do not read the 2.5-perplexity gap between them as decisive.

**Effect of temperature.** Under full training at $\alpha=0.5$, test perplexity was 155.96 at $T=0.5$, 112.60 at $T=1$ and 110.90 at $T=2$. Sharpening the teacher is clearly harmful: $T=0.5$ is about 43 perplexity worse than $T=1$. Softening to $T=2$ gives a nominal 1.7-point improvement, but with one seed we treat $T=1$ and $T=2$ as indistinguishable and do not claim $T=2$ is better. The robust conclusion is one-sided: the optimum lies at or just above $T=1$, and going below 1 costs a great deal. This also revises the impression left by Figure 1, where the short sweep placed the optimum at the edge of its grid; extending below $T=1$ shows the optimum is genuinely interior.

**Ruling out the $T^2$ confound.** Since the $T^2$ correction is imperfect below $T=1$, the poor $T=0.5$ result could in principle have reflected a larger effective step size rather than the sharpened target itself. We tested this by repeating the $T=0.5$ run with the KD term rescaled by $c=\|\nabla\mathcal{L}_{\text{KD}}(T{=}1)\|/\|\nabla\mathcal{L}_{\text{KD}}(T{=}0.5)\|$, recalibrated every epoch on a fixed probe batch. The control reached test perplexity 153.95 against 155.96 for the standard convention -- a 2.0-point difference, roughly $22\times$ smaller than the 43-point penalty of using $T=0.5$ at all. The $T=0.5$ result is therefore a real consequence of sharpening the teacher, not an artifact of gradient scale.

The calibration itself did not behave as the static probe suggested, which is worth reporting. Measured on a cross-entropy-trained student the $T=0.5$ gradient was $3.8\times$ too large, implying $c\approx0.26$; measured *during* KD training, the required correction averaged $c=1.18$ and exceeded 1 in 78\% of epochs, ranging from 0.77 to 1.48. Once the student is actually being trained to match the teacher, its distribution tracks the teacher's closely enough that the gradient mismatch at $T=0.5$ largely disappears. The $T^2$ factor thus works better in practice at $T<1$ than the asymptotic argument predicts -- a null result, but one we could only establish by running the control.

\begin{figure}[t]
\centering
\vspace{-6pt}
\makebox[\linewidth][c]{\includegraphics[width=1.10\linewidth]{../results/temperature_alpha_grid.png}}
\vspace{-8pt}
\caption{Effect of $T$ at $\alpha=0.5$. Left: perplexity against $T$, with the gradient-matched $T=0.5$ control starred. Middle: validation curves. Right: the per-epoch correction $c$, mostly above 1, contradicting the static probe's prediction.}
\end{figure}

# Discussion

Here we answer the five questions and address points that came up around the presentation.

**Q1 — Does distillation help?** Yes. Both teacher-guided students roughly halve the corpus-only test perplexity (90.7 and 115.6 vs.\ 227.2).

**Q2 — Is the teacher alone enough?** Almost, but not quite. Among the three original objectives the teacher-only student was best, which we did not expect at the beginning. Sweeping $\alpha$ shows why and refines the answer: the optimum is interior, at $\alpha=0.25$ (test perplexity 85.59) rather than at $\alpha=0$ (88.06). A light hard-label term helps, while a heavy one is very costly ($\alpha=0.75$ gives 152.44). Our original $\alpha=0.5$ simply sat on the wrong side of the optimum, which is why teacher-only beat it. The overfitting explanation still holds and is now visible across the whole sweep: the best epoch falls monotonically from 50 at $\alpha=0$ to 14 at $\alpha=1$, so hard labels drive the student to memorize the small corpus sooner. The teacher's soft distribution is a gentler, information-rich target that is harder to memorize, which lines up with Hinton et al.'s observation that soft targets act like a regularizer [1].

**Q3 — Which temperature is best?** Our original sweep put the optimum at the edge of its grid ($T=1$), which left open whether an even sharper target would do better. Extending below 1 answers this: at $\alpha=0.5$ under full training, $T=0.5$ reaches only 155.96 against 112.60 at $T=1$ and 110.90 at $T=2$. The optimum is therefore interior, lying at or just above $T=1$, and sharpening is clearly harmful. The capacity argument still explains why very high temperatures fail -- a high temperature asks the student to match the teacher's whole 50,257-way distribution, and a 30M model does not have the capacity for that, so it is better off spending its capacity getting the top token right [1]. But that argument does not extend to $T<1$: a near-one-hot teacher discards the dark knowledge that makes distillation work in the first place, leaving a signal little better than hard labels. A gradient-matched control confirms this is an effect of the target distribution rather than of step size.

**Q4 — How far behind the teacher?** A fair way behind (90.7 vs.\ 29.4), and the gap is actually *understated*: the teacher is judged zero-shot (it was never trained on WikiText-2) while the students train in-domain, so a teacher fine-tuned on WikiText would look even better. This is unsurprising given the $8\times$ smaller Transformer stack.

**Q5 — What do we gain?** 4.1$\times$ fewer parameters, about 4$\times$ smaller on disk, and about 1.8$\times$ faster generation. The speedup is smaller than the parameter ratio because GPT-2 ties the output layer to the token-embedding table, so every generated token still pays for a projection over all 50,257 tokens (the part that only halved), and at batch size 1 generation is limited by latency, not raw compute.


# Conclusions

Our results show that knowledge distillation can greatly improve a smaller GPT-2 model. The Teacher only student achieved a test perplexity of 90.7, followed by Teacher + corpus at 115.6 and Corpus only at 227.2. This means that both methods using teacher guidance performed much better than training only on the true next tokens. Teacher only gave the best result among those three. The temperature sweep selected $T=1$ over the softer values it tested, and the later grid confirmed that $T=1$ is close to optimal while showing that going sharper still, to $T=0.5$, is clearly worse. However, all students still performed worse than the GPT-2 teacher, which achieved a test perplexity of 29.4.

The student also provided a clear efficiency improvement. It used 30.3M parameters instead of 124.4M, required about 119 MB instead of 475 MB on disk, and generated about 1.8$\times$ faster than the teacher. Teacher only continued improving until epoch 39, while Corpus only reached its best result at epoch 13. This suggests that the teacher distribution helped reduce overfitting on the small WikiText-2 dataset. Extending the study with an $\alpha$ and temperature grid sharpened two of these conclusions. Sweeping $\alpha$ showed the best mixture is interior, at $\alpha=0.25$, which improved our best student from a test perplexity of 90.7 to 85.59; a small hard-label term helps, but the corpus signal must stay secondary to the teacher. Extending the temperature axis below 1 showed the optimum is likewise interior: sharpening to $T=0.5$ costs about 43 perplexity, and a gradient-matched control confirmed this reflects the sharpened target itself rather than a change in gradient scale.

Overall, the project shows that distillation can provide a useful balance between model quality, size, and generation speed. Our experiment was limited by one random seed and a small dataset. We addressed the other two original limitations directly: $\alpha$ was swept over five values, and the temperature axis was extended below 1 and re-run under full training rather than a five-epoch budget. A single seed remains the main threat to the finer distinctions -- the 2.5-point gap between $\alpha=0.25$ and $\alpha=0$, and the 1.7-point gap between $T=2$ and $T=1$, are too small to call without repeated runs.

Future work should first repeat the experiments with several random seeds and report the average and variation of the results. This would show whether the differences between the methods are reliable. A full two-dimensional $\alpha\times T$ grid, rather than the two one-dimensional slices we ran, would show whether the best $\alpha$ shifts with temperature. Raising the epoch cap would also settle the $\alpha=0$ versus $\alpha=0.25$ comparison, since the teacher-only student was still improving when training stopped. Training on WikiText-103 would provide more data and may reduce overfitting. A teacher fine-tuned on the same dataset would also give a stronger and fairer upper bound. Testing several student sizes would show how quality, speed, and model size change as student capacity increases. 

# References

[1] G. Hinton, O. Vinyals, and J. Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.

[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108, 2019.

[3] J. Kaplan et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361, 2020.

# Appendix

**A. Parameter breakdown.** How the counts in the Design section split between the embedding tables and the Transformer stack.

: Parameter breakdown (millions).

| Component | Teacher | Student | Ratio |
|---|---:|---:|---:|
| Embedding tables | 39.4 | 19.7 | 2.0$\times$ |
| Transformer stack | 85.1 | 10.6 | 8.0$\times$ |
| **Total** | **124.4** | **30.3** | **4.1$\times$** |

**B. Experimental configuration.** The settings shared by all training runs.

: Experimental configuration.

| Setting | Value | Setting | Value |
|---|---|---|---|
| Dataset | wikitext-2-raw-v1 | Optimizer | AdamW, lr 5e-4 |
| Block size | 1024 | Epochs / patience | 50 / 2 |
| Batch size | 16 | Sweep epochs | 5 |
| Teacher | GPT-2 small (124.4M) | $\alpha$ (main runs) | 0.5 |
| Student | 6L / 384 / 6h (30.3M) | Temperatures | 1,2,4,7,10,15 |
| $\alpha$ grid | 0, 0.25, 0.5, 0.75, 1 at $T=1$ | $T$ grid | 0.5, 1, 2 at $\alpha=0.5$ |
| Grid training | full, early stopping | Grad-matched control | $T=0.5$, $c$ per epoch |

**Grid hardware.** The $\alpha$ and temperature grid ran on a dual AMD Radeon PRO W7900 machine (48 GB per card, ROCm 7.2.4, PyTorch 2.11, transformers 5.13) with one independent training process pinned per GPU. Two concurrent processes showed no measurable contention (0.930 s/step alone versus 0.930 and 0.943 s/step together), giving close to $2\times$ throughput; `DataParallel` across both cards gave only about 8\%, since the 30M student is too small to amortize per-step replication. The eight runs completed in 4.4 hours. Because Table 1 was produced earlier on different hardware, the grid re-trains the three original configurations so all comparisons within that section are internally consistent.

**C. Sample generations (greedy decoding).** The teacher is fluent; the corpus-only student loops tightly; the teacher-guided students are rough but less degenerate.

: Representative greedy generations (truncated).

| Prompt / Model | Continuation |
|---|---|
| **"The city was famous for its"** | |
| Teacher | ...high-speed rail system, which was built in the 1930s. The city's first subway was built in the 1930s... |
| Teacher only | ..."greatest and most beautiful and beautiful, but also a great example of the most beautiful..." |
| Teacher + corpus | ...founding in the city of the city of the city of the city of the city... |
| Corpus only | ...city . The city 's city 's city is a city , and the city 's city ... |
| **"Once upon a time in a small village"** | |
| Teacher | ...a young girl named Nana was kidnapped by a group of bandits. She was taken to a small village... |
| Teacher only | ...and the village's the village's had been evacuated to the village. |
| Teacher + corpus | ...of the village of the village of the village... John 's village. |
| Corpus only | ...The first phase of the area of the area of the area of the area... |
