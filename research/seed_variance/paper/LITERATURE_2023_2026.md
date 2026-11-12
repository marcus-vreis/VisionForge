# Literature 2023-2026 for the seed-variance preprint

Verified 2026-11-12. Method: each entry opened at the arXiv abs page, the arXiv API (`export.arxiv.org`: title, authors, date, journal_ref, comments), Crossref, the JMLR page, the PMLR page or the ACL Anthology page, as listed in the table. Summaries are paraphrased from what the opened page states (abstract level); full texts were NOT read, so no number from inside a paper is used.

Not edited: `main.tex`, `refs.bib`.

## 1. Table

| Key | Reference | ID | URL checked | Why it matters here |
|---|---|---|---|---|
| jordan2023variance | K. Jordan. On the Variance of Neural Network Training with respect to Test Sets and Distributions. arXiv preprint, 2023 (v1 4 Apr 2023, v4 10 Jun 2024). No venue listed. | arXiv:2304.01910 | https://arxiv.org/abs/2304.01910 ; arXiv API | Run-to-run spread on a finite test set is mostly test-set sampling noise, with near-independent per-example errors; briefly covers augmentation and fine-tuning. Frames why small test sets (ours: 1,600 images) inflate apparent seed noise. |
| altintas2025butterfly | G. S. Altintas, D. Kwok, C. Raffel, D. Rolnick. The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions. ICML 2025, PMLR 267:1314-1342. | arXiv:2506.13234 | https://proceedings.mlr.press/v267/altintas25a.html ; https://arxiv.org/abs/2506.13234 ; arXiv API | A tiny perturbation early in training sends otherwise identical runs to different solutions, and the sensitivity fades later; also covers fine-tuning settings. Mechanistic support for seed noise surviving a pretrained start. Author order follows PMLR (arXiv lists Kwok first; first two contributed equally). |
| bui2025seeds | N. T. Bui, G. K. Savova, L. Wang. Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models. IJCNLP-AACL 2025, pp. 41-46. | DOI 10.18653/v1/2025.ijcnlp-short.3 ; arXiv:2503.07329 | https://aclanthology.org/2025.ijcnlp-short.3/ ; arXiv API | Seed choice in fine-tuning changes aggregate scores and also which individual predictions flip; adds a per-prediction consistency measure. Language, not vision, but the same fine-tuning question as Dodge et al. |
| banerjee2024training | S. Banerjee, T. Marrinan, R. Cannon, T. Chiang, A. D. Sarwate. Measuring training variability from stochastic optimization using robust nonparametric testing. arXiv preprint, 2024 (v1 12 Jun 2024, v2 15 Apr 2025). No venue listed. | arXiv:2406.08307 | https://arxiv.org/abs/2406.08307 ; arXiv API | Proposes a nonparametric test of training variability and a way to estimate how many runs are needed; includes a transfer-learning fine-tuning example. Relevant to "how many seeds". |
| mathieu2024adastop | T. Mathieu, M. M. Centa, R. Della Vecchia, H. Kohler, A. Shilova, O.-A. Maillard, P. Preux. AdaStop: Adaptive Statistical Testing for Sound Comparisons of Deep RL Agents. Transactions on Machine Learning Research, 2024. | arXiv:2306.10882 | https://arxiv.org/abs/2306.10882 (journal_ref "TMLR 2024") ; https://mlanthology.org/tmlr/2024/mathieu2024tmlr-adastop | A sequential test that decides how many runs to launch while controlling family-wise error. RL testbed, but the argument that fixed small run counts are unsound transfers. Author order follows the TMLR record (arXiv v1 order differs). Volume/pages not found. |
| du2025pairedbootstrap | W. Du. When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements. arXiv preprint, 24 Nov 2025. No venue listed. | arXiv:2511.19794 | https://arxiv.org/abs/2511.19794 ; arXiv API | Argues single runs and unpaired tests make small gains look significant; proposes paired multi-seed bootstrap. Close to our paired-seed design and reversal rate. Single-author, not peer reviewed as far as the page shows. |
| colas2018seeds | C. Colas, O. Sigaud, P.-Y. Oudeyer. How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv preprint, 2018. | arXiv:1806.08295 | https://arxiv.org/abs/1806.08295 ; arXiv API | Classic: power-analysis view of how many seeds a two-algorithm comparison needs (t-test, bootstrap). Direct ancestor of our "seeds for 80% power". Pre-2023, kept as central classic. |
| benavoli2017time | A. Benavoli, G. Corani, J. Demsar, M. Zaffalon. Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. JMLR 18(77):1-36, 2017. | JMLR v18/16-305 | https://www.jmlr.org/papers/v18/16-305.html | Argues for Bayesian alternatives to null-hypothesis tests when comparing classifiers. Classic; pairs with Demsar 2006 already cited and marks the alternative to our Wilcoxon/Holm choice. |
| brigato2021tune | L. Brigato, B. Barz, L. Iocchi, J. Denzler. Tune It or Don't Use It: Benchmarking Data-Efficient Image Classification. ICCV 2021 Workshops, pp. 1071-1080. | DOI 10.1109/iccvw54120.2021.00125 ; arXiv:2108.13122 | Crossref (title, authors, pages, date 2021-10) ; https://arxiv.org/abs/2108.13122 ; arXiv API | Small-data vision benchmark: a well-tuned plain baseline is hard to beat, so reported gains can be tuning artefacts. Supports caution on ablation claims in the same regime. |
| brigato2022small | L. Brigato, B. Barz, L. Iocchi, J. Denzler. Image Classification With Small Datasets: Overview and Benchmark. IEEE Access 10:49233-49250, 2022. | DOI 10.1109/ACCESS.2022.3172939 ; arXiv:2212.12478 | Crossref (title, authors, volume, pages) ; https://arxiv.org/abs/2212.12478 | Survey plus five-dataset benchmark for image classification with little data; defines the regime. 2022, optional. |
| stevens2025datagap | S. Stevens, S. M. Rayeed, J. Kline. Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications. Imageomics Workshop at NeurIPS 2025 (per arXiv comment); arXiv v1 8 Apr 2025, v2 14 Oct 2025. | arXiv:2504.06486 | https://arxiv.org/abs/2504.06486 ; arXiv API | Argues settings with a few hundred to a few thousand labelled images are under-evaluated and small-data evaluation should be routine. Motivates our regime. Workshop acceptance rests on the arXiv comment only; proceedings page not opened. |

Count: 11 verified. Dated 2023-2026: jordan, altintas, bui, banerjee, mathieu, du, stevens (7; mathieu arXiv 2023, TMLR 2024). Earlier: colas2018, benavoli2017, brigato2021, brigato2022 (4).

Verification level: abstract and metadata only, not full text. bui2025seeds: arXiv v2 says it supersedes an "unauthorized draft" under the same ID, so the ACL version is the one cited. altintas2025butterfly and mathieu2024adastop: author order differs between arXiv and the published record; the published record is used.

## 2. BibTeX

```bibtex
% verificado em https://arxiv.org/abs/2304.01910 em 2026-11-12
@misc{jordan2023variance,
  author = {Jordan, Keller},
  title = {On the Variance of Neural Network Training with respect to Test Sets and Distributions},
  year = {2023},
  eprint = {2304.01910},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

% verificado em https://proceedings.mlr.press/v267/altintas25a.html em 2026-11-12
@inproceedings{altintas2025butterfly,
  author = {Alt{\i}nta{\c{s}}, G{\"u}l Sena and Kwok, Devin and Raffel, Colin and Rolnick, David},
  title = {The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {267},
  pages = {1314--1342},
  year = {2025},
  eprint = {2506.13234},
  archivePrefix = {arXiv}
}

% verificado em https://aclanthology.org/2025.ijcnlp-short.3/ em 2026-11-12
@inproceedings{bui2025seeds,
  author = {Bui, Nghia Tuan and Savova, Guergana K. and Wang, Lijing},
  title = {Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models},
  booktitle = {Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  pages = {41--46},
  year = {2025},
  doi = {10.18653/v1/2025.ijcnlp-short.3}
}

% verificado em https://arxiv.org/abs/2406.08307 em 2026-11-12
@misc{banerjee2024training,
  author = {Banerjee, Sinjini and Marrinan, Tim and Cannon, Reilly and Chiang, Tony and Sarwate, Anand D.},
  title = {Measuring training variability from stochastic optimization using robust nonparametric testing},
  year = {2024},
  eprint = {2406.08307},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML}
}

% verificado em https://arxiv.org/abs/2306.10882 e https://mlanthology.org/tmlr/2024/mathieu2024tmlr-adastop em 2026-11-12
@article{mathieu2024adastop,
  author = {Mathieu, Timoth{\'e}e and Centa, Matheus Medeiros and Della Vecchia, Riccardo and Kohler, Hector and Shilova, Alena and Maillard, Odalric-Ambrym and Preux, Philippe},
  title = {{AdaStop}: Adaptive Statistical Testing for Sound Comparisons of Deep {RL} Agents},
  journal = {Transactions on Machine Learning Research},
  year = {2024},
  eprint = {2306.10882},
  archivePrefix = {arXiv}
}

% verificado em https://arxiv.org/abs/2511.19794 em 2026-11-12
@misc{du2025pairedbootstrap,
  author = {Du, Wenzhang},
  title = {When +1\% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements},
  year = {2025},
  eprint = {2511.19794},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

% verificado em https://arxiv.org/abs/1806.08295 em 2026-11-12
@misc{colas2018seeds,
  author = {Colas, C{\'e}dric and Sigaud, Olivier and Oudeyer, Pierre-Yves},
  title = {How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments},
  year = {2018},
  eprint = {1806.08295},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

% verificado em https://www.jmlr.org/papers/v18/16-305.html em 2026-11-12
@article{benavoli2017time,
  author = {Benavoli, Alessio and Corani, Giorgio and Dem{\v{s}}ar, Janez and Zaffalon, Marco},
  title = {Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis},
  journal = {Journal of Machine Learning Research},
  volume = {18},
  number = {77},
  pages = {1--36},
  year = {2017}
}

% verificado em https://arxiv.org/abs/2108.13122 e Crossref 10.1109/iccvw54120.2021.00125 em 2026-11-12
@inproceedings{brigato2021tune,
  author = {Brigato, Lorenzo and Barz, Bj{\"o}rn and Iocchi, Luca and Denzler, Joachim},
  title = {Tune It or Don't Use It: Benchmarking Data-Efficient Image Classification},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages = {1071--1080},
  year = {2021},
  doi = {10.1109/ICCVW54120.2021.00125}
}

% verificado em https://arxiv.org/abs/2212.12478 e Crossref 10.1109/ACCESS.2022.3172939 em 2026-11-12
@article{brigato2022small,
  author = {Brigato, Lorenzo and Barz, Bj{\"o}rn and Iocchi, Luca and Denzler, Joachim},
  title = {Image Classification With Small Datasets: Overview and Benchmark},
  journal = {IEEE Access},
  volume = {10},
  pages = {49233--49250},
  year = {2022},
  doi = {10.1109/ACCESS.2022.3172939}
}

% verificado em https://arxiv.org/abs/2504.06486 em 2026-11-12
@misc{stevens2025datagap,
  author = {Stevens, Samuel and Rayeed, S M and Kline, Jenna},
  title = {Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications},
  year = {2025},
  note = {Imageomics Workshop at NeurIPS 2025},
  eprint = {2504.06486},
  archivePrefix = {arXiv}
}
```

The `primaryClass` fields come from the arXiv pages opened; for stevens it was not shown, so it is omitted. Page numbers of mathieu2024adastop were not found. Author name encodings (Altintas) should be checked once in the PDF.

## 3. Proposed paragraph (replaces the `\todo` at line ~95)

```latex
Recent work keeps finding that seed noise is not removed by a pretrained start.
\citet{altintas2025butterfly} show that tiny perturbations early in training send
otherwise identical runs to different solutions, and \citet{bui2025seeds} report
seed effects on fine-tuned language models both in aggregate scores and in
which individual predictions change. \citet{jordan2023variance} separates the
variance of a run on a finite test set from variance on the underlying
distribution, which matters when test sets are small. For deciding how many
runs a comparison needs, \citet{colas2018seeds} gave a power-analysis
treatment, \citet{mathieu2024adastop} a sequential test with family-wise error
control, and \citet{banerjee2024training} a nonparametric test of training
variability; \citet{du2025pairedbootstrap} argue that single runs and unpaired
tests make small gains look significant and propose a paired multi-seed
bootstrap, and \citet{benavoli2017time} propose Bayesian alternatives to
null-hypothesis tests. In the small-data regime, \citet{brigato2021tune} found
that a carefully tuned baseline is hard to beat, and \citet{stevens2025datagap}
argue that settings with few labelled images deserve more evaluation. To our
knowledge, none of these measures seed noise against the effect of concrete
ablations in fine-tuned small-data vision, or the rate at which a $k$-seed
experiment reaches the wrong conclusion.
```

Claim check: each sentence restates only what the opened abstract or metadata states. The last sentence is a novelty claim resting on the searches below, hence "To our knowledge".

## 4. Searched and discarded

- Picard 2021 (arXiv:2109.08203): already in refs.bib.
- arXiv:2208.10758 and arXiv:2011.15124 (vision-and-language): seen in snippets only, off regime.
- arXiv:2609.15706 and arXiv:2609.11310 (VLM papers with seed studies): snippets only, not opened, off regime.
- Deep Dominance / ASO (Dror et al.) and the `deepsig` package: snippets only, NLP-oriented, no primary page opened. Candidate if ASO is wanted.
- Henderson et al., "Deep RL that matters": RL, pre-2023, not opened; Colas and Mathieu cover the RL side.
- Brigato, "A Close Look at Deep Learning with Small Data" (arXiv:2003.12843): 2020, snippets only, redundant with brigato2021tune/2022.
- Springer review of ML with small data (s40537-025-01346-9) and ScienceDirect small-organ segmentation benchmark (S1746809426017805): snippets only; generic review / medical segmentation.
- Small-ViT architecture papers (arXiv:2402.03752, 2501.06220, 2501.06040): about CIFAR architectures, not variance or evaluation.
- arXiv:2206.01161, a Frontiers bioengineering ViT paper, arXiv:2308.12366, H2RBench, Local SGD hits: seed variation incidental or off topic.
- Not found: a 2023-2026 paper measuring seed noise against concrete ablation effects, or reversal rates of k-seed experiments, in small-data pretrained CNN fine-tuning. Only a general web search tool was used (not Google Scholar or Semantic Scholar), so absence is not proof.
