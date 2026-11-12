# Many Models, One Forecast: a hands-on workshop on ensemble methods

An **ensemble** combines several models into one prediction. In this workshop you learn why that works, and you build
the six-model ensemble that gave our team its best score in the sea-surface temperature forecasting competition.

This folder is self-contained: copy it anywhere and it runs on its own, on a laptop CPU.

## The project need

Our forecasting networks are noisy. Retrain the same network with a new random seed and its validation score moves by up
to 0.006 °C, which is more than the gap between our two best architectures. A final submission should not depend on a
lucky seed. Averaging several models fixed that and lowered our Kaggle error by 2.6% (0.27419 vs 0.28150), with no new
model and no extra tuning.

## What you will learn

By the end you can:

1. Explain why averaging models works, and when it doesn't (the error correlation ρ).
2. Build a seed + architecture ensemble from trained models, and check it step by step.
3. Describe bagging, boosting and stacking, and pick the right ensemble method for your own project.

## What's in the folder

```
ensemble-workshop/
├── README.md                          # this file: setup, tutorial, expected results, troubleshooting
├── requirements.txt
├── sst_models.py                      # the two network architectures, checkpoint loading and the metric
├── checkpoints/                       # six trained networks: 2 architectures × 3 seeds
├── data/validation.npz                # competition validation split (728 examples, 2022–2023)
├── notebooks/
│   ├── 01_tutorial_ocean_ensemble.ipynb   # START HERE: the hands-on tutorial
│   └── 02_other_ensemble_methods.ipynb    # then: bagging, boosting, stacking in scikit-learn
└── slides/ensemble-workshop.pptx      # the slides
```

## Setup

You need **Python 3.12** (3.10 or 3.11 also work; 3.13 does not support `torch==2.2.2`). Run these in a terminal,
inside the `ensemble-workshop` folder:

```bash
# macOS and Linux
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # downloads PyTorch, about 200 MB
jupyter lab                            # or open the notebook in VS Code and pick the .venv kernel
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

On Windows, if PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once and try again.

Then open `notebooks/01_tutorial_ocean_ensemble.ipynb` and run the cells from top to bottom with **Shift + Enter**.
The notebooks are saved without outputs: run each step and compare your result with its ✅ checkpoint.

## Tutorial (notebook 01)

The notebook walks through these steps. After each one, compare your output with the checkpoint.

Every score in notebook 01 is printed for each forecast day (day 1, day 2, day 3) and as their mean, which is the
competition score.

| Step | What you do | Why it matters | ✅ Checkpoint |
|---|---|---|---|
| 0 | Setup check: import the libraries and find the checkpoints | Confirms your environment before anything else | `torch 2.2.2, 6 checkpoints found`, `Setup OK` |
| 1 | Load the validation data and score the persistence baseline | Shapes of the input and target; the score every model must beat | `X: (728, 14, 12, 14)`, `y: (728, 3, 12, 14)`, persistence days `0.1792 / 0.3171 / 0.4262`, mean `0.3075` |
| 2 | Run the six trained networks | Shows seed luck, and that each architecture is better on a different forecast day | six forecasts of shape `(728, 3, 12, 14)`, means `0.2747` to `0.2805`; spatial wins day 1, flattened wins day 3 |
| 3 | **Deep ensemble:** average the 3 seeds of each architecture | Averaging cancels run-to-run luck | `tcn × 3 seeds` mean `0.2738`, `spatial_tcn × 3 seeds` mean `0.2743` |
| 4 | **Averaging:** combine the two architectures 50/50 | Different models make different mistakes, here on different days | `all six, 50/50` days `0.1474 / 0.2770 / 0.3811`, mean `0.2685` |
| 5 | Measure the error correlation ρ between the models | Explains why step 4 gained more than step 3 | `0.963` same architecture, `0.898` different |
| 6 | **Your turn** (below) | Use the method on a new question | your score and its rank out of 20 |

Shapes: `X` is `(examples, 14 input days, latitude, longitude)` and `y` is `(examples, 3 forecast days, latitude, longitude)`.
The score is the RMSE in °C over the 143 ocean cells, averaged over the 3 forecast days (lower is better).

Results can differ in the 4th decimal on another CPU or library version. Within ±0.0005 counts as a match.

## Your turn: you can afford three trainings

**Scenario.** Your team has compute for only three training runs. Which three of the six models would you ensemble?

1. In step 6 of the notebook, edit `my_pick` to any 3 of `("tcn", 444)`, `("tcn", 445)`, `("tcn", 446)`,

   `("spatial_tcn", 444)`, `("spatial_tcn", 445)` and `("spatial_tcn", 446)`, then run the cell.
2. Run the check cell under it. It prints your score, its rank among all 20 possible 3-model ensembles, and whether your

   pick mixes both architectures.
3. Try at least two picks, including one that mixes both architectures.
4. Write down what changed, and why. Use ρ from step 5.

**How to check your result.** Picks from one architecture only score about `0.274` (rank 19 or 20 of 20). Picks that mix
both architectures score between `0.2685` and `0.2714`. Every mixed pick beats both one-architecture picks, because
models built differently share fewer of their mistakes (ρ ≈ 0.90 instead of 0.96).

Stretch: replace `np.mean` with `np.median`, or try a 2-model ensemble.

## The other ensemble methods (notebook 02)

`notebooks/02_other_ensemble_methods.ipynb` shows the classic families our project did not use, with scikit-learn on
small toy datasets. Every cell runs in seconds.

| Step | Method | The idea | ✅ Checkpoint (cross-validated accuracy) |
|---|---|---|---|
| 1 | Why averaging works | The variance formula, simulated | the dots sit on the lines; curves flatten at ρ |
| 2 | Voting | Different models vote, hard or soft | soft voting `0.890`, average member `0.882` |
| 3 | **Bagging**, random forests | Same model, random resamples of the data | one deep tree `0.836`, random forest `0.885` |
| 4 | **Boosting** | Weak models, each fixing the previous errors | one stump `0.798`, gradient boosting `0.887` |
| 5 | **Stacking** | A meta-model learns whom to trust | stacking `0.870` beats every member; done wrong `0.767`, done right `0.891` |

**Your turn (step 3):** change `n_estimators` in the `my_forest` cell to 1, 5, 20 and 300. Expect about `0.83`, `0.87`,
`0.88` and `0.89`: accuracy rises quickly and then levels off, because the trees share part of their errors, the
same limit as our seeds. Accuracies can differ in the 3rd decimal on another machine.

## When to use it, and its limits

| If your model… | Try |
|---|---|
| is a neural network that scores differently every run | **Deep ensemble**: average 3 to 5 seeds |
| is one of several good but different models | **Averaging** (voting); stacking only if you can cross-validate |
| is a deep decision tree that overfits | **Bagging**: a random forest |
| is too simple and underfits | **Boosting**: XGBoost, LightGBM |

Limitations and tradeoffs:

* **Cost.** M models take M× the training and prediction time.
* **Near-copies add nothing.** If the members' errors are almost identical (ρ close to 1), you pay for M models and get one.
* **Honest validation.** Fix the weights in advance (equal weights are a strong default) or cross-validate them. Tuning
them on the set you report makes the score optimistic.
* **Every member should be good on its own.** A weak member drags the average down.
* **Spread is not calibrated uncertainty.** Members that share errors are overconfident together (bonus section B5).

## Adapt it to your final project

```python
preds = []
for seed in [0, 1, 2]:
    model = train_model(train_data, seed=seed)  # your code
    preds.append(model.predict(X_val))
ensemble = np.mean(preds, axis=0)
# then add a second, different architecture the same way
```

With scikit-learn models, `VotingRegressor` and `VotingClassifier` average different models for you. Checklist: report
the mean ± spread over seeds rather than one run; average 3 to 5 seeds; add a second, different model; fix the weights
before you look at the validation score; check the error correlation before adding a member.

## Going further (optional)

* **Bonus sections B1–B5** at the end of the tutorial notebook: lucky pairs, all 63 subsets, tuning the weight honestly,
the ambiguity decomposition, and where on the map the ensemble helps.
* **More to try in notebook 02**: change the label noise, drop the neural net from the stack, or set
`max_features=None` in the random forest (which turns it back into plain bagging).

## Troubleshooting

| You see | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or `sst_models`) | The notebook runs on a different Python than your `.venv` | In Jupyter or VS Code, choose the `.venv` kernel; or start `jupyter lab` from a terminal where the `.venv` is activated |
| `AssertionError: Checkpoints not found` or `FileNotFoundError: ../data/validation.npz` | The notebook's working folder is not `notebooks/` | Start Jupyter inside `ensemble-workshop` and open the notebook from the file browser, so paths like `../checkpoints` resolve |
| `ERROR: No matching distribution found for torch==2.2.2` | Python 3.13 or newer | Create the `.venv` with Python 3.12, or install a newer torch without the version pin |
| `RuntimeError: Numpy is not available` | NumPy 2 installed next to torch 2.2 | `pip install "numpy<2"`, then restart the kernel |
| Step 2 seems stuck | Running six networks takes about 30 seconds on a CPU | Wait for the six lines to print |
| Your numbers differ in the 4th decimal | Different CPU or library versions | Fine: within ±0.0005 counts as a match |

Still stuck: read the full error, find the first cell that failed, and check its inputs (paths, the kernel, the shapes
printed in step 1) before changing anything else.

## Key numbers (validation RMSE, °C)

| Model | Mean | Day 1 / 2 / 3 |
|---|---|---|
| Persistence | 0.3075 | 0.1792 / 0.3171 / 0.4262 |
| Single network (mean of 6 runs) | 0.2780 | 0.1542 / 0.2869 / 0.3928 |
| Flattened TCN × 3 seeds | 0.2738 | 0.1522 / 0.2824 / 0.3868 |
| Spatial TCN × 3 seeds | 0.2743 | 0.1497 / 0.2835 / 0.3899 |
| **All six, 50/50** | **0.2685** | 0.1474 / 0.2770 / 0.3811 |

Kaggle public score: 0.27419 for the ensemble, vs 0.28150 for the best single TCN.

## Sources and credits

* Data: EGR 444 sea-surface temperature forecasting competition (validation split, 2022–2023).
* Models, checkpoints and notebooks: our team's competition code.
* [scikit-learn user guide: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html).
* Lakshminarayanan, Pritzel & Blundell, [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474), NeurIPS 2017.
* Krogh & Vedelsby, *Neural Network Ensembles, Cross Validation, and Active Learning*, NIPS 7, 1995.
* Hastie, Tibshirani & Friedman, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), chapters 8, 10, 15 and 16.
* Breiman, *Bagging Predictors*, Machine Learning, 1996 · Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics, 2001 · Wolpert, *Stacked Generalization*, Neural Networks, 1992.
* Francis Galton, *Vox Populi*, Nature, 1907 (the ox-weighing story).
* Libraries: PyTorch, NumPy, pandas, scikit-learn, Matplotlib, Jupyter.
