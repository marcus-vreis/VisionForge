# Seed variance study — verified literature
Verified on October 15, 2026 by web search; items marked NOT VERIFIED must be checked by hand.

## Picard 2021
- **Citation (verified):** David Picard. "Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision". arXiv preprint arXiv:2109.08203 [cs.CV], 2021. DOI: 10.48550/arXiv.2109.08203.
- **Source checked:** https://arxiv.org/abs/2109.08203
- **What it measured:** Scanned up to $10^4$ random seeds on CIFAR-10 and fewer seeds on ImageNet using standard computer vision architectures to observe test accuracy distributions. Found that while the overall variance is relatively modest, outlier seeds ("black swans") frequently achieve performance differences large enough to establish or invalidate state-of-the-art claims under single-seed reporting.
- **What this study adds:** Evaluates seed variance specifically in transfer learning with ImageNet-pretrained backbones under data-scarce regimes (10% subsets), testing paired seed designs and small ablation effects with VisionForge.
- **Allowed wording:** Random seed selection in computer vision models can produce substantial performance variations and outlier results that alter benchmark rankings or mask true differences when using single-seed evaluations.
- **Corrections:** The recalled title was slightly abbreviated ("torch.manual_seed(3407) is all you need..." omitting the subtitle ": On the influence of random seeds in deep learning architectures for computer vision"); it is an arXiv preprint rather than a peer-reviewed conference paper.

## Bouthillier et al. 2021
- **Citation (verified):** Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Nazanin Mohammadi Sepahvand, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Dmitriy Serdyuk, Tal Arbel, Chris Pal, Gaël Varoquaux, Pascal Vincent. "Accounting for Variance in Machine Learning Benchmarks". Proceedings of Machine Learning and Systems (MLSys), Vol. 3, pp. 747–769, 2021. arXiv:2103.03098.
- **Source checked:** https://arxiv.org/abs/2103.03098
- **What it measured:** Modeled the empirical benchmarking pipeline across five deep-learning tasks to quantify error rates in detecting true algorithmic improvements, isolating variance contributions from data sampling, parameter initialization, and hyperparameter tuning. Demonstrated that uncontrolled random variations frequently explain published performance deltas and that sampling multiple sources of variation reduces estimator error.
- **What this study adds:** Extends benchmark variance analysis to small-sample vision regimes (10% data subsets) in fine-tuning scenarios, evaluating paired-seed hypothesis tests and providing the VisionForge tool for automated CI and Holm correction.
- **Allowed wording:** Empirical comparisons in machine learning benchmarks suffer from high error rates due to variance from random seeds, data splits, and hyperparameters, often causing reported algorithmic improvements to be statistical artifacts unless multiple trials and variance sources are accounted for.
- **Corrections:** none.

## Summers & Dinneen 2021
- **Citation (verified):** Cecilia Summers, Michael J. Dinneen. "Nondeterminism and Instability in Neural Network Optimization". Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:9913–9922, 2021. arXiv:2103.04514.
- **Source checked:** https://proceedings.mlr.press/v139/summers21a.html
- **What it measured:** Established an experimental protocol isolating multiple sources of nondeterminism (GPU/CUDA operations, data ordering, parameter initialization) in neural network optimization, showing that all sources yield similar degrees of model diversity due to chaotic training instability where even a single-bit difference in parameters diverges trajectories.
- **What this study adds:** Investigates whether deterministic training and paired random seeds can control this instability during fine-tuning of pretrained vision models across 10% data subsets and small ablations using VisionForge.
- **Allowed wording:** Deep neural network optimization exhibits end-to-end training instability where minute stochastic perturbations or nondeterminism cause optimization paths to diverge to distinct local minima with variable performance.
- **Corrections:** none.

## Dodge et al. 2020
- **Citation (verified):** Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, Noah A. Smith. "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping". arXiv preprint arXiv:2002.06305 [cs.CL], 2020. DOI: 10.48550/arXiv.2002.06305.
- **Source checked:** https://arxiv.org/abs/2002.06305
- **What it measured:** Fine-tuned BERT across hundreds of random seeds on four GLUE benchmark tasks, isolating and comparing the variance introduced by task-head weight initialization versus training data order. Found that both contribute comparably to downstream performance variance and that fine-tuning frequently diverges partway through training on small datasets.
- **What this study adds:** Translates the examination of fine-tuning seed sensitivity and small-dataset instability from NLP language models to transfer learning with computer vision backbones across full and 10% data subsets using VisionForge.
- **Allowed wording:** Pretrained model fine-tuning is sensitive to random seed selection—with weight initialization and data ordering contributing comparably to performance variance—and displays high instability and divergence on small datasets.
- **Corrections:** The recalled description provided an informal topic summary ("fine-tuning, weight initialization and data order (NLP)") rather than the exact paper title, and did not note that it is an arXiv preprint.

## Gundersen et al. 2022
- **Citation (verified):** Odd Erik Gundersen, Kevin Coakley, Christine Kirkpatrick, Yolanda Gil. "Sources of Irreproducibility in Machine Learning: A Review". arXiv preprint arXiv:2204.07610 [cs.LG], 2022. DOI: 10.48550/arXiv.2204.07610.
- **Source checked:** https://arxiv.org/abs/2204.07610
- **What it measured:** Synthesized literature on reproducibility in machine learning into a comprehensive theoretical taxonomy mapped across the scientific method, categorizing failure modes into study design, algorithmic stochasticity (including random seeds), implementation factors, observation, evaluation, and documentation.
- **What this study adds:** Directly addresses implementation and evaluation irreproducibility factors by deploying VisionForge to automate deterministic multi-seed training, paired statistical inference, and Holm-corrected reporting in small-sample CV research.
- **Allowed wording:** Irreproducibility in machine learning stems from a structured spectrum of factors across the experimental lifecycle, with algorithmic randomness and experiment design choices frequently undermining validity.
- **Corrections:** The recalled description ("review of sources of irreproducibility in ML") was a descriptive phrase; the exact published preprint title is "Sources of Irreproducibility in Machine Learning: A Review".

## Demšar 2006
- **Citation (verified):** Janez Demšar. "Statistical Comparisons of Classifiers over Multiple Data Sets". Journal of Machine Learning Research (JMLR), 7(1):1–30, 2006.
- **Source checked:** https://www.jmlr.org/papers/v7/demsar06a.html
- **What it measured:** Analyzed theoretical and empirical properties of parametric and non-parametric statistical tests for comparing multiple classifiers over multiple datasets, showing why standard paired $t$-tests violate independent-sample assumptions and recommending the Wilcoxon signed-ranks test, Friedman test, and Critical Difference diagrams.
- **What this study adds:** Applies non-parametric and paired statistical testing principles directly to deep learning multi-seed replicates and ablation effects in VisionForge, verifying whether multi-seed conclusions remain statistically reliable.
- **Allowed wording:** Statistical comparisons of classifiers across benchmarks should rely on safe, robust non-parametric tests (such as Wilcoxon signed-ranks and Friedman tests) rather than naive parametric tests that violate independence assumptions.
- **Corrections:** The recalled title was abbreviated as "statistical comparisons of classifiers (JMLR)"; the exact title is "Statistical Comparisons of Classifiers over Multiple Data Sets".

## Nadeau & Bengio 2003
- **Citation (verified):** Claude Nadeau, Yoshua Bengio. "Inference for the Generalization Error". Machine Learning, 52(3):239–281, 2003. DOI: 10.1023/A:1024068626366.
- **Source checked:** https://link.springer.com/article/10.1023/A:1024068626366
- **What it measured:** Formulated a theoretical investigation into the variance of cross-validation estimators of generalization error, proving that variance estimators based solely on standard cross-validation are biased due to sample overlap and leading to the corrected resampled $t$-test that maintains nominal Type I error rates.
- **What this study adds:** Extends the principles of accounting for non-independent sample variance to deep learning seed-induced variance, where shared data subsets and paired initialization seeds introduce covariance between model runs.
- **Allowed wording:** Conventional variance estimators in machine learning comparisons underestimate true generalization variance by ignoring correlations induced by training sample overlap, requiring corrected variance formulations to prevent inflated false-positive rates.
- **Corrections:** none.

## Brooke 1996
- **Citation (verified):** John Brooke. "SUS: A 'Quick and Dirty' Usability Scale". In P. W. Jordan, B. Thomas, B. A. Weerdmeester, & I. L. McClelland (Eds.), Usability Evaluation in Industry, Chapter 35, pp. 189–194 (electronic pp. 207–212), Taylor & Francis / CRC Press, 1996. DOI: 10.1201/9781498710411-35. ISBN: 9780748404605.
- **Source checked:** https://www.taylorfrancis.com/chapters/edit/10.1201/9781498710411-35/sus-quick-dirty-usability-scale-john-brooke
- **What it measured:** Developed and validated the System Usability Scale (SUS), a 10-item Likert-scale questionnaire generating a standardized score from 0 to 100 to evaluate the subjective usability of systems and tools across diverse industrial applications.
- **What this study adds:** Uses the SUS framework to empirically measure the usability and practitioner workflow efficiency of the VisionForge GUI and CLI when executing paired multi-seed training and statistical evaluations.
- **Allowed wording:** The System Usability Scale (SUS) is an established, technology-independent 10-item instrument for quantifying subjective usability and user satisfaction with software interfaces.
- **Corrections:** The recalled description ("Brooke (1996), the System Usability Scale") did not cite the full chapter title ("SUS: A 'Quick and Dirty' Usability Scale") or the edited volume (*Usability Evaluation in Industry*).

## Banerjee et al. 2024
- **Citation (verified):** Sinjini Banerjee, Tim Marrinan, Reilly Cannon, Tony Chiang, Anand D. Sarwate. "Measuring training variability from stochastic optimization using robust nonparametric testing". arXiv preprint arXiv:2406.08307 [stat.ML], 2024. DOI: 10.48550/arXiv.2406.08307.
- **Source checked:** https://arxiv.org/abs/2406.08307
- **What it measured:** Formulated a robust nonparametric testing framework using the $\alpha$-trimming level to measure neural network model similarity and variability across stochastic optimization runs and random seeds, estimating the number of training runs required to form a statistically representative ensemble and demonstrating its utility in transfer learning fine-tuning.
- **What this study adds:** Investigates paired multi-seed replicates and hypothesis testing for small-sample computer vision architectures in VisionForge, comparing seed variance against specific ablation effect sizes.
- **Allowed wording:** Standard summary metrics like test accuracy obscure the substantial model variability introduced by random training seeds, which can be quantified and bounded using robust nonparametric hypothesis testing to determine adequate sample sizes of training runs.
- **Corrections:** none.

## Eryılmaz et al. 2024
- **Citation (verified):** Bahadır Eryılmaz, Osman Alperen Koraş, Jörg Schlötterer, Christin Seifert. "Investigating the Impact of Randomness on Reproducibility in Computer Vision: A Study on Applications in Civil Engineering and Medicine". arXiv preprint arXiv:2410.02806 [cs.CV], 2024. DOI: 10.48550/arXiv.2410.02806.
- **Source checked:** https://arxiv.org/abs/2410.02806
- **What it measured:** Investigated the impact of CUDA-induced randomness and stochasticity across benchmark and real-world image datasets, demonstrating that hardware- and execution-level randomness can account for performance score variations up to 4.77% and analyzing the performance trade-offs of enforcing deterministic execution.
- **What this study adds:** Enforces full determinism to isolate pure algorithmic and seed-induced variance in fine-tuning ImageNet models on small datasets (including a 10% regime) across paired ablation tests with VisionForge.
- **Allowed wording:** Uncontrolled CUDA and hardware randomness can cause performance fluctuations of up to nearly 5% in computer vision tasks, making deterministic controls essential for reproducible benchmarking.
- **Corrections:** none.

## Sharma 2025
- **Citation (verified):** Udit Sharma. "When Does Pairing Seeds Reduce Variance? Evidence from a Multi-Agent Economic Simulation". arXiv preprint arXiv:2512.24145 [cs.LG], 2025. DOI: 10.48550/arXiv.2512.24145.
- **Source checked:** https://arxiv.org/abs/2512.24145
- **What it measured:** Analyzed the statistical properties of comparative evaluation under shared/paired random seeds, proving mathematically and empirically that evaluating competing systems on identical seeds induces matched stochastic realisations that yield strict variance reduction, tighter confidence intervals, and higher statistical power whenever seed-level outcomes correlate positively.
- **What this study adds:** Applies paired-seed variance reduction directly to deep vision model comparisons and ablation testing on 10% data subsets, implementing automated paired hypothesis tests in VisionForge.
- **Allowed wording:** Evaluating competing stochastic machine learning pipelines using paired, shared random seeds reduces comparative variance and increases statistical power relative to independent seed draws.
- **Corrections:** none.

## Bui et al. 2025
- **Citation (verified):** Nghia Bui, Guergana Savova, Lijing Wang. "Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models". arXiv preprint arXiv:2503.07329 [cs.CL], 2025. DOI: 10.48550/arXiv.2503.07329.
- **Source checked:** https://arxiv.org/abs/2503.07329
- **What it measured:** Evaluated the influence of random seeds on pretrained model fine-tuning across GLUE and SuperGLUE benchmarks, analyzing macro-level variance (accuracy and F1 score spread) and introducing a micro-level consistency metric to prove that individual sample predictions fluctuate significantly across seeds even when macro metrics appear identical.
- **What this study adds:** Extends the study of fine-tuning seed sensitivity to small-data vision regimes, analyzing whether macro effect sizes and seed variance interact across 10% data splits and CNN/ViT architectures in VisionForge.
- **Allowed wording:** Fine-tuning pretrained models across different random seeds causes significant variance at both macro (aggregate metrics) and micro (instance-level prediction consistency) levels.
- **Corrections:** none.

## Sumit et al. 2026
- **Citation (verified):** Dipto Sumit, Sakib Ul Haque, Farig Sadeque. "Below the Noise Floor: Bimodal Seed Collapse and Distinct Failure Modes in Small-Model Knowledge Distillation". arXiv preprint arXiv:2608.27729 [cs.CL], August 2026. DOI: 10.48550/arXiv.2608.27729.
- **Source checked:** https://arxiv.org/abs/2608.27729
- **What it measured:** Evaluated multi-seed stability across eight knowledge distillation variants for a 1.5B student model, finding per-seed standard deviations ranging from 2.8 to 48.7 percentage points (swallowing claimed gains under 5 points), bimodal seed collapse in multiple variants, and demonstrating that an apparent +3.78 pp gain on a single seed reversed to -2.70 pp upon multi-seed evaluation.
- **What this study adds:** Explores whether seed variance similarly swallows small ablation gains (H1) and causes conclusion reversals (H2) during transfer learning of computer vision backbones on small datasets, using paired seeds in VisionForge.
- **Allowed wording:** Run-to-run seed variance can exceed reported algorithmic gains (placing improvements below the noise floor) and cause apparent single-seed performance advantages to completely reverse under multi-seed evaluation.
- **Corrections:** none.

## Le & Cao 2026
- **Citation (verified):** Minh Le, Phuong Cao. "On the Extreme Variance of Certified Local Robustness Across Model Seeds". arXiv preprint arXiv:2601.13303 [cs.LG], January 2026. DOI: 10.48550/arXiv.2601.13303.
- **Source checked:** https://arxiv.org/abs/2601.13303
- **What it measured:** Evaluated the impact of training random seeds on neural network verification, demonstrating that models differing only in random seeds exhibit extreme variance in certified robustness—with standard deviations statistically exceeding reported marginal improvements—and that "lucky" seeds fail to generalize to new test sets.
- **What this study adds:** Evaluates whether seed standard deviation similarly dominates ablation effect sizes (H1) in vision fine-tuning under low data, testing paired Holm-corrected comparisons in VisionForge.
- **Allowed wording:** Random seed variation during neural network training introduces variance that can be statistically larger than marginal published improvements, demonstrating that single-seed findings cannot guarantee reliable generalization.
- **Corrections:** none.

## Does existing work already answer the question?
**Partially.**

Existing literature partially answers the question by establishing that seed variance often matches or exceeds small effect sizes, but leaves the specific small-data computer vision fine-tuning regime unmeasured. Regarding hypothesis H1, Picard (2021), Summers & Dinneen (2021), Le & Cao (2026), and Sumit et al. (2026) demonstrate that run-to-run seed variance routinely swallows marginal improvements, though none quantify the ratio of seed standard deviation to ablation effect size across ImageNet transfer cells. Regarding hypothesis H2, Bouthillier et al. (2021) and Sumit et al. (2026) document frequent ranking changes and conclusion reversals under single-seed draws, while Sharma (2025) proves the theoretical variance reduction of paired seeding, but neither measures empirical inversion rates for fine-tuned vision models. Regarding hypothesis H3, Dodge et al. (2020) observed heightened divergence and variance on small datasets in NLP fine-tuning, but existing vision work has not evaluated whether data scarcity universally inflates seed standard deviation across differing vision architectures. Furthermore, prior vision seed sweeps like Picard (2021) trained models from scratch or examined single datasets without enforcing strict GPU determinism. To ensure the study provides unique empirical value, the question should be narrowed to quantifying the minimum paired-seed threshold ($k$) required to suppress conclusion inversions below 10% specifically during ImageNet transfer learning on 10% data subsets. This sharpens the contribution to validating whether architectural inductive biases (CNNs vs. ViTs) moderate low-data seed variance under deterministic paired evaluation in VisionForge.

## Review notes (coordinator, 2026-10-15)
- Drafted by Gemini 3.8 Flash (agy) with web search; reviewed here. The six 2024–2026
  items were re-checked against the arXiv API (`export.arxiv.org/api/query?id_list=…`):
  every id resolves, and titles and authors match the entries above.
- The eight recalled items match their known bibliographic records; page ranges and
  DOIs still get checked once more when they go into `paper.bib`.
- To check by hand before citing: the synthesis says Picard (2021) "trained models from
  scratch or examined single datasets". His ImageNet experiments used *pretrained*
  networks, so the gap this study fills is the small-data regime, pairing and
  determinism, not pretraining itself. Reword that sentence accordingly.
- Verdict "Partially" stands. The suggested narrowing (the minimum paired-seed k that
  keeps inversions under 10% in the 10% regime, and whether CNN vs. ViT moderates
  seed variance) is a decision for the protocol (Task B2), not made here.
