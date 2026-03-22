&nbsp;
&nbsp;
<p align="center">
  <img width="800" src="./figures/conecta_logo.png" />
</p> 

&nbsp;

# AutoTSLM: Reasoning over OBD-II Telemetry for Driver Behavior Analysis using LLMs

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Raspberry Pi](https://img.shields.io/badge/Edge-Raspberry%20Pi-C51A4A.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

### ✍🏾 Authors: [Morsinaldo Medeiros](https://github.com/Morsinaldo), [Marianne Silva](https://github.com/MarianneDiniz), [Dennis Brandão](https://scholar.google.com/citations?user=OxSKwvEAAAAJ&hl=pt-BR), and [Ivanovitch Silva](https://github.com/ivanovitchm)


## 📄 Abstract / Overview

Understanding driving behavior from vehicle telemetry remains a challenging task due to the temporal complexity of sensor signals and the limited interpretability of conventional machine learning approaches. Recent advances in large language models have shown strong capabilities for reasoning and explanation generation, opening new possibilities for analyzing time-series data. In this work, we investigate the use of Time-Series Language Models (TSLMs) for reasoning over automotive telemetry and propose *AutoTSLM*, a framework that adapts the OpenTSLM paradigm to process multivariate OBD-II time-series windows. The proposed approach combines a temporal encoder with pretrained language models through multimodal alignment mechanisms, including Soft Prompt conditioning and Flamingo-style cross-attention, enabling the model to generate both driver behavior classifications and natural-language explanations grounded in telemetry signals. Experimental results show that explicitly integrating time-series representations substantially improves performance compared with text-only prompting baselines. Furthermore, deployment experiments on a Raspberry Pi demonstrate that lightweight model configurations can provide a practical balance between reasoning capability and computational efficiency for edge inference.



## 📂 Repository Structure

```text
METROAUTOMOTIVE2026-AutoTSLM/
├── data/                               # OBD-II datasets, plots, baselines, and evaluation outputs
│   ├── obd_cot_gpt5.jsonl              # Main OBD-II-CoT dataset
│   ├── obd_alignment_new_data.jsonl    # Alignment dataset
│   ├── obd_alignment_small.jsonl       # Small debug dataset
│   ├── obd_cot_plots/                  # Rendered temporal plots
│   ├── obd_baselines/                  # Baseline outputs
│   └── llm_judge_outputs/              # LLM judge artifacts
├── notebooks/                          # Training and analysis notebooks
│   ├── obd_soft_prompt_training.ipynb
│   ├── obd_flamingo_alignment_training.ipynb
│   ├── obd_article_plots.ipynb
│   └── ...
├── results/                            # Training summaries, plots, and inference outputs
├── scripts/                            # Dataset generation, plotting, and inference scripts
├── src/opentslm/                       # Local OpenTSLM-based implementation
├── src/open_flamingo/                  # Local OpenFlamingo dependency
├── requirements.training.txt
├── requirements.rpi.inference.txt
└── README.md
```

## ⚙️ Environment Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

For training, notebooks, and analysis:

```bash
pip install -r requirements.training.txt
```

For lightweight inference, especially on Raspberry Pi:

```bash
pip install -r requirements.rpi.inference.txt
```

Run commands from the repository root so imports resolve correctly.


## 🧰 Recommended Scripts

These are the scripts that should be treated as the main maintained entry points:

- `scripts/generate_obd_cot_dataset.py`: builds windows, renders temporal plots, and optionally generates CoT rationales.
- `scripts/generate_obd_alignment_dataset.py`: builds the alignment dataset from raw telemetry windows.
- `scripts/generate_obd_article_plots.py`: regenerates the consolidated article plots and summary tables from saved outputs.
- `scripts/run_obd_soft_prompt_inference.py`: runs Soft Prompt inference and stores metrics/system traces.
- `scripts/run_obd_flamingo_inference.py`: runs Flamingo inference and stores metrics/system traces.


## 🧪 Data Generation

### 1. Generate temporal plots from raw OBD-II CSV files

Use `scripts/generate_obd_cot_dataset.py` to segment raw telemetry into 120-sample windows and render the four-panel temporal plots (`Speed`, `RPM`, `EngineLoad`, `ThrottlePos`).

```bash
python scripts/generate_obd_cot_dataset.py \
  --input-dir path/to/raw_obd_csvs \
  --plot-dir data/obd_cot_plots \
  --output-jsonl data/obd_cot_preview.jsonl \
  --window-size 120 \
  --stride 120 \
  --samples-per-file 2 \
  --allow-missing
```

This command already produces the plot images under `data/obd_cot_plots/`.

### 2. Generate the OBD-II-CoT dataset

The same script can call an OpenAI model to generate the rationale/explanation field for each window. The main dataset tracked in the repository is `data/obd_cot_gpt5.jsonl`.

```bash
python scripts/generate_obd_cot_dataset.py \
  --input-dir path/to/raw_obd_csvs \
  --plot-dir data/obd_cot_plots \
  --output-jsonl data/obd_cot_gpt5.jsonl \
  --window-size 120 \
  --stride 120 \
  --samples-per-file 2 \
  --max-samples 195 \
  --allow-missing \
  --use-openai \
  --openai-model gpt-5
```

Expected outputs:

- `data/obd_cot_gpt5.jsonl`
- `data/obd_cot_plots/*.png`

### 3. Generate the alignment dataset

Use `scripts/generate_obd_alignment_dataset.py` when you need the text-alignment dataset used by the Flamingo-style experiments.

```bash
python scripts/generate_obd_alignment_dataset.py \
  --input-dir path/to/raw_obd_csvs \
  --output-jsonl data/obd_alignment_new_data.jsonl \
  --output-csv data/obd_alignment_new_data.csv \
  --samples-per-file 4 \
  --window-size 24 \
  --stride 24 \
  --allow-missing
```

If you want Ollama-generated text instead of the deterministic fallback templates:

```bash
python scripts/generate_obd_alignment_dataset.py \
  --input-dir path/to/raw_obd_csvs \
  --output-jsonl data/obd_alignment_new_data.jsonl \
  --output-csv data/obd_alignment_new_data.csv \
  --samples-per-file 4 \
  --window-size 24 \
  --stride 24 \
  --allow-missing \
  --use-ollama \
  --model gemma3:12b
```

## 🧠 Training

Training is currently notebook-driven.

### Soft Prompt

Run [notebooks/obd_soft_prompt_training.ipynb](./notebooks/obd_soft_prompt_training.ipynb).

Current notebook behavior:

- loads `data/obd_cot_gpt5.jsonl`
- trains the Soft Prompt model
- saves the best checkpoint to `data/obd_sp_best.pt`
- writes training/inference summaries to `results/notebook_metrics/`

### Flamingo

Run [notebooks/obd_flamingo_alignment_training.ipynb](./notebooks/obd_flamingo_alignment_training.ipynb).

Current notebook behavior:

- loads `data/obd_cot_gpt5.jsonl`
- trains the Flamingo variant
- saves the best checkpoint to `data/obd_flamingo_best.pt`
- writes training/inference summaries to `results/notebook_metrics/`


## 🚀 Inference

The two inference scripts below expect the notebook-generated checkpoints by default.

### Soft Prompt inference

```bash
python scripts/run_obd_soft_prompt_inference.py \
  --dataset data/obd_cot_gpt5.jsonl \
  --checkpoint data/obd_sp_best.pt \
  --llm-id google/gemma-3-270m \
  --device cpu \
  --llm-dtype float32 \
  --split test \
  --offline
```

### Flamingo inference

```bash
python scripts/run_obd_flamingo_inference.py \
  --dataset data/obd_cot_gpt5.jsonl \
  --checkpoint data/obd_flamingo_best.pt \
  --llm-id meta-llama/Llama-3.2-1B \
  --device cpu \
  --llm-dtype float32 \
  --split test \
  --offline
```

Each inference run creates a timestamped folder under `results/raspberry_pi_inference/` with:

- `predictions.jsonl`
- `detailed_metrics.jsonl`
- `system_metrics.csv`
- `system_summary.json`
- `summary.json`

## 📊 Article Plots and Summaries

To rebuild the consolidated plots and tables from the saved experiment artifacts:

```bash
python scripts/generate_obd_article_plots.py
```

Main outputs:

- `results/article_plots/`
- `results/article_tables/obd_model_approach_metrics.csv`
- `results/article_tables/obd_model_approach_metrics.md`


---

## 🗂️ Main Datasets Already Present

- `data/obd_cot_gpt5.jsonl`: main OBD-II-CoT dataset.
- `data/obd_alignment_new_data.jsonl`: alignment dataset.
- `data/obd_alignment_small.jsonl`: reduced debug split.
- `data/obd_cot_plots/`: pre-rendered temporal plots.


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## 👥 About Us

The **Conect2AI** research group is composed of undergraduate and graduate students from the **Federal University of Rio Grande do Norte (UFRN)**. The group focuses on applying Artificial Intelligence and Machine Learning to emerging areas such as **Embedded Intelligence, Internet of Things, and Intelligent Transportation Systems**, contributing to energy efficiency and sustainable mobility solutions.

Website: http://conect2ai.dca.ufrn.br
