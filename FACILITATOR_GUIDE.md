# Facilitator guide: Many Models, One Forecast

For the presenters only: Damian and Cyril. The slides, the README and the notebooks are written for classmates;
everything about *running* the session lives here: who presents what, timing, speaker notes, checkpoints, debugging,
likely questions, and the checklist for the course requirements.

**Framing.** A workplace training session: our team learned a method (ensembling) that made our forecasts reliable, and
we teach our coworkers to use it on the project. The class plays the coworkers.

**Goal.** Every classmate leaves with a working ensemble, can explain why it worked (ρ), knows the other ensemble
methods, and can repeat the process on their own project.

---

## 1. Who presents what

| | Damian | Cyril |
|---|---|---|
| **Focus** | What we did in the forecasting competition, and the two methods behind it: deep ensembles (seeds) and averaging | Ensemble methods in general: why they work, and the other methods: bagging, boosting, stacking |
| **Slides** | 1–4 (project need and result), 9–17 (tutorial and Your turn), 25 (adapt it to your project) | 5–8 (key idea and the toolbox), 18–22 (other methods), 23–24 (which method when, limitations), 26 (sources) |
| **Notebook** | `01_tutorial_ocean_ensemble.ipynb`, steps 0–6 | `02_other_ensemble_methods.ipynb`, steps 3–5 and its Your turn |
| **When not presenting** | Roams during Cyril's notebook 02 | Roams during the whole tutorial and Your turn (the busiest debugging time) |
| **Together** | Slide 27 (takeaways and questions): Damian takes project questions, Cyril takes method questions | |

Why this split works: the session has two threads, *our project* and *the wider toolbox*. Each presenter owns one
thread, one notebook and the questions about it, and each one is the helper in the room while the other leads a
hands-on part. Cyril opens the theory (slides 5–8) so the class hears both voices in the first ten minutes and so
the toolbox overview on slide 8 is introduced by the person who later teaches the other methods.

Both presenters must still be able to explain every step and fix every row of the troubleshooting table.

## 2. Run of show (40 minutes + 5 buffer)

| Time | Part | Slides | Notebook | Lead | Roam |
|---|---|---|---|---|---|
| Before 0:00 | People arrive: start the install | 9 on screen | | Damian | Cyril |
| 0:00–3:00 | Intro: project need, finished result, agenda | 1–4 | | **Damian** | |
| 3:00–7:30 | Key idea: ox poll, formula, the toolbox | 5–8 | | **Cyril** | Damian checks installs |
| 7:30–9:30 | Checkpoint 0: setup check | 9–10 | 01, step 0 | **Damian** | Cyril |
| 9:30–22:00 | Tutorial: steps 1–5 and the two-knobs recap | 11–16 | 01, steps 1–5 | **Damian** | Cyril |
| 22:00–27:00 | Your turn: three trainings, two people share | 17 | 01, step 6 | **Damian** | Cyril |
| 27:00–35:00 | Other methods: bias vs variance, bagging, boosting, stacking, then run them | 18–22 | 02, steps 3–5 | **Cyril** | Damian |
| 35:00–38:00 | Use it: which method when, limitations, adapt it | 23–25 | | **Cyril** (23–24), **Damian** (25) | |
| 38:00–40:00 | Sources, takeaways, questions | 26–27 | | both | |
| 40:00–45:00 | Buffer: stragglers, extra questions, bonus sections | | 01, B1–B5 | both | both |

If you fall behind: shorten the ox poll to one minute; on slide 22 Cyril runs notebook 02 as a demo instead of
waiting for everyone; summarize slide 24 in one sentence. Never cut checkpoint 0 or the Your turn on slide 17.

## 3. Speaker notes

Slide numbers refer to `slides/ensemble-workshop.pptx`. These are talking points, not a script to read out.

### Damian · Intro (slides 1–4)

**Before 0:00, slide 9 on screen.** "While you settle in, open a terminal in the ensemble-workshop folder and run
these four commands. The install takes a few minutes, so start it now." Then go back to slide 1.

**1 · Cover.** "For the next 40 minutes you are our teammates on the ocean-forecasting project. Cyril and I found a
method that made our forecasts more reliable, and we want everyone on the team to be able to use it." Ensembling means
combining several models into one forecast.

**2 · The project need.** The problem: retrain the same network, change only the random seed, and the score moves by
up to 0.006 °C. That is more than the gap between our two best architectures, so one run is a lucky draw. The
method: average several models. The result: our best Kaggle score, 2.6% lower error than our best single model, with
no new training.

**3 · What you will build.** "This chart is what you will reproduce yourself." Walk down the bars: persistence, a
typical network, three seeds of one network, two networks × three seeds. Read the three outcomes.

**4 · Today.** Five parts, two notebooks. "I take you through what we did in the competition, in notebook 01. Cyril
shows you why it works and the other ensemble methods, in notebook 02." Both notebooks are saved with outputs, so
anyone who falls behind can still follow. Hand over: "Cyril, why does this work at all?"

### Cyril · Key idea (slides 5–8)

**5 · Ox poll.** 20 seconds to write a guess. Ask two or three people out loud. "Will the crowd's middle guess beat
yours?"

**6 · The answer.** Median 1,207 lb, actual 1,198 lb: under 1% off, better than almost every individual, including
the cattle experts. The condition that matters: people guessed *independently*, so their errors pointed in different
directions and cancelled.

**7 · The formula.** M models, each with error variance σ², and ρ is how similar their mistakes are. More models
shrink only the second term; ρσ² never goes away. Ten independent models keep 10% of the variance, ten models with
ρ = 0.7 keep 73%, ten copies keep 100%. "So the goal is good models that make different mistakes. Damian will measure
ρ on our real models in step 5."

**8 · The toolbox.** Five classic ways to get models that disagree. "The top two are what our project uses, and
Damian walks you through them next. The bottom three, bagging, boosting and stacking, are mine: we come back to them
in part 4." Hand over to Damian for the setup check.

### Damian · Tutorial and Your turn (slides 9–17, notebook 01)

Cyril roams the room from here until slide 18.

**9 · Step 0, setup check.** Everyone runs step 0. "Hands up when you see `Setup OK`." Cyril goes straight to anyone
without a hand up. Move on when most of the room is green; the rest can follow on the saved outputs and catch up
during Your turn.

**10 · Troubleshooting.** Leave it up while people fix their setup. The usual ones: wrong kernel, wrong folder,
Python 3.13, NumPy 2.

**11 · Step 1, load the data.** "This is the competition task." Input: 14 daily sea-temperature maps on a 12 × 14
grid; output: the next 3 days; score: RMSE over the 143 ocean cells. Read the shapes out loud: examples, days,
latitude, longitude. Persistence, "tomorrow looks like today", scores 0.3075 and is surprisingly hard to beat; our
networks predict a correction to it. **Checkpoint 1.**

**12 · Step 2, six networks.** Our two finalists after a lot of grid searches. The flattened TCN reads each day's map
as one long vector, so it sees the whole box. The spatial TCN treats each day like an image, then runs one small TCN
per grid cell, with a third of the parameters. We trained each three times and changed only the seed. Scores run
from 0.2747 to 0.2805: that spread is seed luck. If asked: seed 446 is best for both because we tuned every
hyperparameter with it; the honest single-model score is about 0.278. **Checkpoint 2.**

**13 · Step 3, deep ensemble.** "First method we used." The seed decides the starting weights, the batch order and
the dropout masks, so each run lands on a different, equally good solution and makes slightly different mistakes.
`np.mean(..., axis=0)` averages three forecasts cell by cell. A single run is about 0.278; three seeds give 0.2738
and 0.2743, about 0.004 better for free. **Checkpoint 3.**

**14 · Step 4, averaging.** "Second method." Average the two seed-averages 50/50. For numbers, voting is just the
mean; for classes you would count votes or average probabilities. We fixed 50/50 before looking at validation, so the
score stays honest. Result: 0.2685, better than every single run on every forecast day, and 0.27419 on Kaggle.
**Checkpoint 4.**

**15 · Step 5, why mixing wins.** "This is Cyril's formula on our data." ρ between seeds of the same network: 0.963.
Between the two architectures: 0.898. Lower ρ means a bigger gain, which is why step 4 gained more than step 3. All
the correlations are high because most of the error is weather that no model saw coming; no ensemble removes that.
**Checkpoint 5.**

**16 · Two knobs.** Down is more seeds, which removes luck. Right is more architectures, which adds diversity. Ask
the room: "Which knob gained more?" (Architectures.)

**17 · Your turn.** Read the scenario: compute for only three trainings. Give four minutes; Damian and Cyril both
roam. Then ask two people: what did you pick, what did you get, why? Expected: one-architecture picks ≈ 0.274
(rank 19–20 of 20); mixed picks 0.2685–0.2714. The answer we want to hear: mixing architectures lowers ρ, so the
average removes more error with the same number of models. Early finishers: median instead of mean, or two models.
Hand over: "That is what worked for neural networks. Cyril, what if your model is not a neural network?"

### Cyril · The other methods (slides 18–22, notebook 02)

Damian roams from here.

**18 · Bias vs variance.** Think of a model as someone throwing darts. Bias is bad aim: the throws land together
but in the wrong place, because the model is too simple. Variance is shaky hands: the aim is right on average, but
every throw lands somewhere else. The navy diamond is the average of the shaky throws, and it sits on the bullseye:
that is what Damian's averaging did. "Averaging fixes variance. It does nothing for bias. The next three methods
cover both cases."

**19 · Bagging.** Take one shaky model, usually a deep decision tree. Train many copies, each on a random sample of
the rows drawn with replacement, then vote or average. Each tree sees about 63% of the rows; the rest give a free
test score. A random forest also hides some features at each split so the trees differ even more. It is one line in
scikit-learn. Why we did not use it: our networks need all the data, and new seeds already made them differ.

**20 · Boosting.** The relay team. Train weak models one after another; each new one fits what the team so far still
gets wrong. The formula: new ensemble = old ensemble + a small step η times the new model. XGBoost and LightGBM are
this, and they still win most table-data competitions. Watch out: it can chase noise, so use small steps and early
stopping. Why we did not use it: boosting fixes bias, and our problem was variance.

**21 · Stacking.** The manager. Instead of a fixed average, a small second model learns how much to trust each base
model. The key trick: train the manager on out-of-fold predictions, made by models that never saw that row. Skip
that and the manager trusts whichever model memorized the most. Why we did not use it: with one validation set, a
learned manager would overfit it; tuning our one weight gave 0.50 anyway.

**22 · Notebook 02.** "Open `02_other_ensemble_methods.ipynb` and run it top to bottom; it takes seconds." Scroll to
step 3, 4 and 5 and point at the checkpoints: one deep tree 0.836 → forest 0.885; one stump 0.798 → gradient
boosting 0.887; stacking 0.870 beats its best member 0.853; the trap, 0.767 wrong vs 0.891 right. Then the task:
change `n_estimators` to 1, 5, 20, 300. Ask: "When does adding trees stop helping?" Expected: about 0.83, 0.87,
0.88, 0.89. Link back: the trees share part of their errors, the same limit Damian measured for seeds.

### Cyril, then Damian · Use it (slides 23–27)

**23 · Which method when (Cyril).** Top two rows are what Damian showed; bottom two are what I showed. Shaky
models → average them; too-simple models → boost them; several good but different models → average first, stack
only if you can cross-validate.

**24 · Limitations (Cyril).** Cost scales with the number of models. Near-copies add cost and no accuracy. Never
tune weights on the set you report. A weak member drags the mean down. The spread between members is overconfident.

**25 · Adapt it (Damian).** "This is all it took in our project." The recipe works with any model that has a seed;
`VotingRegressor` does it for scikit-learn models. Walk the checklist; reporting mean ± spread over seeds is worth
doing even if you never ensemble.

**26 · Learn more (Cyril).** Point at the scikit-learn guide and the Deep Ensembles paper; credit the competition
data and our own code.

**27 · Takeaways and questions (both).** Damian reads the first two boxes, Cyril the last two. Damian answers
questions about the project and the tutorial; Cyril answers questions about the methods.

## 4. Running a checkpoint

1. Say what everyone should see, and point at the checkpoint box.
2. Ask for hands up when it matches.
3. Wait until most of the room is green; the roamer goes to anyone who is not.
4. Say what a reasonable result looks like: within ±0.0005 of the slide is a match.
5. Move on; people who are behind can read the saved outputs and catch up.

## 5. Debugging playbook

When someone is stuck:

1. Ask what they expected to see.
2. Read the full error together, from the bottom line up.
3. Find the **first** cell that failed; later errors are usually consequences.
4. Check the environment (which kernel? does `import torch` work?) and the inputs (working folder, shapes printed
   in step 1), then change one thing at a time.
5. Explain the fix so they can do it again, and note it for the README update.

Most likely problems, in order: wrong kernel (`ModuleNotFoundError`), notebook opened from the wrong folder (paths
with `../`), Python 3.13 (no `torch==2.2.2` wheel), NumPy 2 next to torch 2.2 (`Numpy is not available`), slow Wi-Fi
during `pip install` (let them follow the saved outputs and install later). Fixes are in the README troubleshooting
table. Notebook 02 needs only scikit-learn, so it works even where the torch install failed.

## 6. Likely questions

Damian (project):

* **Is 0.005 °C meaningful?** About 2% of the error, consistent on every forecast day and in 88% of ocean cells
  (bonus B5), and it moved us up the leaderboard.
* **Why seeds 444–446?** Arbitrary; 446 was our tuning seed, and 444 and 445 were added for the ensemble.
* **How many seeds?** Gains flatten fast because seeds think alike (ρ ≈ 0.96); 3 to 5 is typical. A different
  architecture is worth more than extra seeds.
* **What does it cost?** Six trainings instead of one. Prediction is still fast: all six run in about 30 s on a CPU.
* **Does the ensemble give uncertainty?** The spread is a hint, but overconfident: about 0.06 °C of spread against
  0.23 °C of error (bonus B5).

Cyril (methods):

* **Why not stacking for the project?** One validation set; learning weights on it would overfit. Tuning the single
  weight on 2022 and testing on 2023 picked 0.50 anyway (bonus B3).
* **Bagging vs a deep ensemble?** Both average copies of one model. Bagging gets its variety from resampling the
  data; a deep ensemble gets it from the random seed and keeps all the data.
* **Can you boost neural networks?** You can, but it is slow (sequential training) and it targets bias; networks
  usually suffer from variance.
* **Hard or soft voting?** Soft when the probabilities are trustworthy: a confident model counts more.
* **Is ensembling cheating on Kaggle?** No: it is standard practice and most winning solutions use it.

## 7. How the materials cover the grading criteria

| Criterion (points) | Where it is covered |
|---|---|
| Project relevance and scope (3) | Slides 2–3 (project need and finished result); a real competition task; everything runs in under a minute on a CPU, no training |
| Technical accuracy and clarity (4) | Numbered steps with the code on the slide and a ✅ checkpoint with expected output and shapes; assumes only course material |
| Hands-on learning and debugging (4) | The class runs every step; two Your turn tasks (slide 17, slide 22) with a way to check; one presenter always roams; troubleshooting slide 10 |
| Preparation and team handoff (4) | README with use case, outcomes, setup, tutorial, expected results, troubleshooting, credits; notebooks saved with outputs; both presenters lead a hands-on part; contribution note |

Course guideline steps → materials: **state the outcome** (slides 3–4), **clear starting point** (slide 9, README
Setup), **one small example in short steps** (slides 11–15), **checkpoints** (every step slide, ✅ cells in both
notebooks), **let the class use it** (slides 17 and 22), **handoff** (slides 23–26: when useful, limitations, adapt,
documentation and credits).

## 8. Before class checklist

- [ ] Topic approved by Dr. Chiovaro (email at least a week before the first workshop session).
- [ ] Replace `[Team names] · [Date]` on the cover slide.
- [ ] **Fresh-clone test:** clone the repo to a new location and follow only the README; reach every checkpoint in
      both notebooks. Cyril does notebook 01 and Damian does notebook 02, so each tests the part the other built.
- [ ] Rehearse the full session, including both Your turn tasks, in **40 minutes or less**. If it runs over, cut
      content (see section 2), not the hands-on parts.
- [ ] Both presenters can explain every step and fix every row of the README troubleshooting table.
- [ ] **By 2 pm on the workshop day:** send Dr. Chiovaro the repo link (or a zip) and the contribution note.
- [ ] Ask classmates to install beforehand if possible: the PyTorch download is about 200 MB. Bring the folder on a
      USB stick as a backup.
- [ ] **Within two days after the workshop:** update the README with any problem you saw and its fix.

## 9. Contribution note (for Dr. Chiovaro)

Edit the preparation lines so they say what each of you actually did, then send it with the materials.

```
Workshop: Many Models, One Forecast (ensemble methods)
Group members: Damian, Cyril

Damian
  Preparation: [e.g. trained the seed checkpoints, built the tutorial notebook (01), wrote the setup and troubleshooting sections]
  Workshop role: led the intro, the tutorial (steps 0–5), the Your turn task and "adapt it to your project";
                 helped classmates debug during notebook 02

Cyril
  Preparation: [e.g. built the other-methods notebook (02) and its slides, ran the fresh-clone test]
  Workshop role: led the key idea (why ensembles work), the other methods (bagging, boosting, stacking) with
                 notebook 02, "which method when" and limitations; helped classmates debug during the tutorial
```

## 10. Maintaining the materials

* The deck is one file, `slides/ensemble-workshop.pptx`; edit it in place. It has no speaker notes on purpose: they
  are in section 3 of this guide.
* If you change a notebook's code, keep the code on the matching slide identical, and update the checkpoint numbers
  in the notebook, the slide and the README together.
* Re-run both notebooks from top to bottom before sharing, so the saved outputs match the code.
