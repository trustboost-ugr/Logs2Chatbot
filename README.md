# Logs2Chatbot: From Dialogue Logs to LLM-Initialized Rasa Bots

This repository contains the code and released Rasa projects used in the paper  
“Automating the Initial Development of Intent-based Task-Oriented Dialog Systems Using Large Language Models: Experiences and Challenges”.

We share:

- The **generation scripts** used to go from transcribed dialogues to initial Rasa 3.x projects (generated artifacts).
- The **harness code** for Phase II, which iteratively probes each bot that validates (therefore it passed Phase I) and applies small, reproducible “micro-patches”.
- The **final bootstrapped bots** (Rasa projects) for four LLMs × two corpora:
  - **Corpora:** “Let’s Go” (English bus information) and **MEDIA** (French hotel booking).
  - **LLMs:** GPT-4o, Claude Opus, Gemini 2.5 Pro, Mistral Small.

> In the paper, Phase I turns noisy LLM-generated YAML into trainable Rasa bots;  
> Phase II uses a scripted probe harness to push them to “passably working” task completion.

---

## Repository layout

At a high level, the repository is organized as follows:

- `projects/`  
  Final Rasa 3.x projects for each *(corpus, LLM, phase)* combination.  
  Each project is a standard Rasa project with `domain.yml`, `data/`, `rules.yml`, `stories.yml`, etc.,
  and can be used directly with `rasa train` / `rasa shell`.

- `generate/`  
  Scripts for **Phase I**:
  - Read dialogue logs.
  - Call LLMs to synthesize Rasa artifacts (domain, NLU, rules/forms, stories).
  - Write out initial Rasa projects under `projects/`.
  These scripts assume you have LLM access (OpenAI, Anthropic, etc.) and the environment from `environment.yml`.

- `phase2-harness/`  
  Scripts for **Phase II**:
  - A unified harness (`measure_kpis.py`) that:
    - Starts a Rasa project on a free port.
    - Runs 50 scripted probes.
    - Logs trackers and produces Completion / Turns / Help-recovery KPIs.
  - Support code for parsing logs and computing summary tables.

- `requirements.txt`  
  Python dependencies for the **Rasa side** (evaluation, running projects, harness without LLM calls).

- `environment.yml`  
  Conda environment for the **LLM side** (generation, LangChain, OpenAI/Anthropic clients, etc.).

> Important: `requirements.txt` and `environment.yml` correspond to **two different environments**.
> They are not meant to be installed into the same virtualenv.

---

## Environments

### 1. Rasa / evaluation environment (Phase I & Phase II Rasa runs)

Use this if you just want to:

- Train and inspect the released bots.
- Run the harness (`measure_kpis.py`) and reproduce the KPI tables.
- Recompute NLU metrics (`rasa test nlu`).

```bash
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
````

This environment is aligned with the Rasa 3.x version we used in the paper.

### 2. LLM / generation environment (Phase I generation only)

Use this if you want to **regenerate** projects from logs with LLMs:

```bash
conda env create -f environment.yml
conda activate logs2chatbot
```

This environment pins versions for:

* LangChain
* OpenAI / Anthropic / other LLM SDKs
* Supporting tooling used in the generation scripts

> Do **not** mix `requirements.txt` into this conda env: keep the Rasa and LLM environments separate.

---

## 3. Working with the released Rasa projects

Each project under `projects/` is a standard Rasa 3.x project. In the Rasa environment:

```bash
# From the project root (where config.yml, domain.yml, data/ live)
cd projects/letsgo_gpt4o_phase2        # example directory; see actual names under projects/
rasa data validate --debug
rasa train --debug

# Launch an interactive shell
rasa shell
```

You can also re-run NLU cross-validation as in the paper:

```bash
rasa test nlu --cross-validation --folds 5 --out results/nlu
```

---

## 4. Running the harness on a project

The harness lives in `phase2-harness/` and is implemented in `measure_kpis.py`.
It:

1. Starts a given Rasa project on a free port.
2. Runs 50 scripted probes (with every 5th probe injecting a `help` detour).
3. Stores conversation logs and trackers.
4. Prints summary KPIs:

   * **Completion %** (probes that reach a goal state).
   * **Median turns** from first user turn to goal.
   * **Help recovery %** (success rate on probes with a help detour).

### 4.1. Point the harness to your Rasa project

Open `phase2-harness/measure_kpis.py` and adjust the paths / config block at the top so that it knows:

* Where your Rasa project lives (root directory with `config.yml`, `domain.yml`, `data/`, etc.).
* Where to write logs (a local `logs/` directory is fine).
* Which corpus/model label to use for tagging runs.

These are simple Python constants; no YAML config is required.

### 4.2. Run the harness

In the **Rasa environment**:

```bash
# From the repo root:
python phase2-harness/measure_kpis.py
```

The script will:

* Start the Rasa server for the configured project.
* Emit 50 probe conversations.
* Write JSON logs (one per dialogue) and a small summary table to disk.

The metrics it prints correspond directly to the **probe-based KPIs** reported in the paper (Completion, Turns, Help-recovery).

---

## 5. Regenerating Rasa projects from dialogue logs

Scripts under `generate/` implement the **Logs → LLM → Rasa YAML** pipeline used in the paper:

* Read raw dialogues (e.g., MEDIA / Let’s Go transcriptions).
* Prompt an LLM to propose:

  * A domain schema (`domain.yml`).
  * NLU examples (`data/nlu.yml`).
  * Rules/forms (`rules.yml`).
  * Stories / conversation flows (`data/stories.yml`).
* Write the resulting artifacts into a new subdirectory under `projects/`.

Typical usage in the **LLM environment**:

```bash
conda activate logs2chatbot
cd generate

# Each script has its own CLI; inspect help:
python <some_generation_script>.py --help

# Example pattern:
# python generate_gpt4o.py --input data/letsgo_dialogues.csv --outdir ../projects/letsgo_gpt4o/original
```

The exact script names and arguments are documented in the docstrings and `--help` of the files in `generate/`.
You will need to set the appropriate API keys (OpenAI, Anthropic, etc.) as environment variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

---

## Reproducing the paper’s results (high-level)

In brief, to replicate the main steps of the paper:

1. **Phase I (optional, regeneration)**

   * Use the scripts in `generate/` (LLM env) to regenerate initial Rasa projects from logs.
   * Switch to the Rasa env and run:

     ```bash
     rasa data validate --debug
     rasa train --debug
     ```

     until each project trains and runs cleanly.
   * Run `rasa test nlu --cross-validation` to obtain accuracy / macro-F1 and per-intent F1.

2. **Phase II (harness bootstrapping)**

   * In the Rasa env, configure `phase2-harness/measure_kpis.py` for your project.
   * Run:

     ```bash
     python phase2-harness/measure_kpis.py
     ```
   * Inspect the logs, apply small YAML “micro-patches” (forms/rules, slot mappings, custom action stubs, lookups), and re-run the harness until reaching the desired KPI thresholds.
   * The final projects in `projects/` correspond to the Phase II state reported in the paper.

3. **Analysis / tables**

   * Use the harness logs and Rasa NLU reports to compute:

     * Overall accuracy & macro-F1.
     * Per-intent F1.
     * Robustness by slice (short tokens, route codes, temporal/date-ish expressions, amenities).
     * Probe-based KPIs (completion, turns, help-recovery).
---

## Citation

If you use this repository, the released bots, or the harness in your work, please cite the Logs2Chatbot paper (bibtex to be added once the final version is available).

---

## Acknowledgements

This code and the released projects were developed as part of the **TrustBoost** project at the University of Granada.
We thank the Rasa open-source community and the maintainers of the Let’s Go and MEDIA corpora for making this line of work possible.
