# Comparative Analysis of Ferroelectric Models

A simulation framework for comparing two ferroelectric device models — **Exponential** and **Tanh** — embedded as synapses in a convolutional spiking neural network (CSNN) trained via **Voltage-Dependent Synaptic Plasticity (VDSP)**. The network is trained unsupervised and evaluated on MNIST / Fashion-MNIST using a downstream SVM classifier. The framework also quantifies how device non-idealities — **device-to-device (D2D)** and **cycle-to-cycle (C2C) variations** — impact classification accuracy.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quick Start — Reproduce the Paper Results](#4-quick-start--reproduce-the-paper-results)
5. [User Guide: Testing Your Own Device Data](#5-user-guide-testing-your-own-device-data)
   - 5.1 [Preparing Your Device Data File](#51-preparing-your-device-data-file)
   - 5.2 [Choosing a Synapse Model](#52-choosing-a-synapse-model)
   - 5.3 [Running Device Characterization](#53-running-device-characterization)
   - 5.4 [Configuring Device Variation Parameters](#54-configuring-device-variation-parameters)
   - 5.5 [Training the CSNN with Your Device](#55-training-the-csnn-with-your-device)
   - 5.6 [Co-Design Algorithm (Exponential Model Only)](#56-co-design-algorithm-exponential-model-only)
   - 5.7 [Evaluating D2D and C2C Robustness](#57-evaluating-d2d-and-c2c-robustness)
6. [Parameter Reference](#6-parameter-reference)
7. [Workflow Diagram](#7-workflow-diagram)
8. [Troubleshooting](#8-troubleshooting)
9. [Expected Results](#9-expected-results)

---

## 1. Project Overview

This framework implements a two-stage pipeline:

**Stage 1 — Unsupervised CSNN Training (VDSP)**  
A single-layer convolutional SNN is trained without labels using voltage-dependent synaptic plasticity. Weight updates are computed directly from device physics: the synapse model translates the neuron membrane potential into a write voltage, which drives the memristive weight update according to the characterised device model.

**Stage 2 — Supervised Readout (Linear SVM)**  
After VDSP training, the network's learned spike representations are extracted as features and fed to a Linear SVM for classification on MNIST or Fashion-MNIST.

Two device (synapse) models are supported:

| Model Name | Description |
|---|---|
| `Ferroelectric` | Exponential switching model — fits a ΔW vs. V curve with an exponential window function |
| `Ferroelectric_Tanh` | Tanh resistance-envelope model — fits upper/lower R vs. V envelopes from pulse-switching measurements |

Both models support D2D and C2C variation injection via configurable noise coefficients.

---

## 2. Repository Structure

```
.
├── config.py              ← Global switches: SYNAPSE_MODEL, TIMESTEPS, device
├── Training.py            ← Entry point: train CSNN + SVM, evaluate accuracy
├── Model.py               ← CSNN_Layerwise: architecture, SVM wrapper
├── Layers.py              ← CsnnLayer and SnnPooling implementations
├── Synapse_Models.py      ← Ferroelectric and Ferroelectric_Tanh synapse classes
├── Characterization.py    ← Curve-fitting of raw device data → model parameters
├── Solver.py              ← Co-design EM algorithm (Exponential model)
├── utils.py               ← DoG transform, helper functions
├── D2D_ploting.py         ← Reproduce D2D variation figures (Fig. 4.7 & 4.8)
├── C2C_plotting.py        ← Reproduce C2C variation figures
├── data/                  ← Place your raw device data here (*.dat)
│   └── ABS_03_summary.dat ← Example device data (included)
└── figures/               ← Output directory for all saved plots
```

---

## 3. Installation

### Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)
- See `requirements.txt` for pinned versions

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/LinyueWeng/Comparative-Analysis-of-Ferroelectric-Models.git
cd Comparative-Analysis-of-Ferroelectric-Models

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create output directories
mkdir figures
```

> **Note on PyTorch version:** `requirements.txt` pins `torch==2.10.0+cu130` (CUDA 13). If you are on CPU only or a different CUDA version, install PyTorch manually from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

---

## 4. Quick Start — Reproduce the Paper Results

To run the full pipeline with the included example device data:

```bash
python Training.py
```

This will:
1. Automatically download **Fashion-MNIST** to `./data/` on first run.
2. Run device characterisation on `data/ABS_03_summary.dat` (or load cached parameters if already computed).
3. Train the CSNN for 1 VDSP epoch until convergence.
4. Extract spike features and train a Linear SVM.
5. Print test accuracy on the Fashion-MNIST test set.

Expected output (Fashion-MNIST, `Ferroelectric_Tanh` model, no variation):
```
>>> Test Accuracy: ~88.3%
```

To switch to MNIST, open `Training.py` and comment/uncomment:
```python
# Fashion-MNIST (default):
full_dataset = datasets.FashionMNIST(...)
test_dataset = datasets.FashionMNIST(...)

# MNIST (uncomment these two lines and comment the two above):
# full_dataset = datasets.MNIST(...)
# test_dataset = datasets.MNIST(...)
```

---

## 5. User Guide: Testing Your Own Device Data

This section walks you through the complete workflow for plugging in your own memristor measurement data and evaluating how your device performs in the network.

---

### 5.1 Preparing Your Device Data File

Place your raw measurement file inside the `data/` directory.

**Update the path in `Characterization.py`, line 16:**
```python
RAW_DEVICE_DATA_PATH = "data/YOUR_FILE.dat"   # ← change this
```

#### Required file format

The framework loads your file using `pandas.read_csv`. The file must be **comma-separated** (`.dat` or `.csv`) with a **header row** as the first line.

The included example file `data/ABS_03_summary.dat` illustrates the exact format:

```
pulseAmplitude,deltaRneg(measured at -80mV),RnegInitial,deltaRpos(measured at +80mV),RposInitial
-2.393,-9233422.556,940707223.691,-5027733.684,1113622949.542
-1.066,12987171.376,931473801.135,2016120.153,1108595215.857
1.321,399087810.065,944460972.511,468431606.063,1110611336.010
...
```

#### Column descriptions

| Column | Unit | Description |
|---|---|---|
| `pulseAmplitude` | V | Write pulse voltage applied to the device. Typically ranges from −3 V to +3 V. Positive values drive LTD (depression); negative values drive LTP (potentiation). |
| `deltaRneg(measured at -80mV)` | Ω | Change in resistance measured at −80 mV read bias after the write pulse. A negative value means resistance decreased (potentiation). |
| `RnegInitial` | Ω | Resistance state of the device **before** the write pulse, measured at −80 mV. Typical range: ~9×10⁸ Ω (low state) to ~3×10⁹ Ω (high state). |
| `deltaRpos(measured at +80mV)` | Ω | Change in resistance measured at +80 mV read bias after the write pulse. |
| `RposInitial` | Ω | Resistance state of the device **before** the write pulse, measured at +80 mV. |

> **Two read-bias columns:** The device is characterised at both −80 mV and +80 mV read bias. `Characterization.py` uses both sets of measurements to fit a more robust model of the switching envelope. Both column pairs (`deltaRneg`/`RnegInitial` and `deltaRpos`/`RposInitial`) are required.

#### Adapting your own measurement data

If your measurement setup uses different column names, you have two options:

**Option A — Rename your columns** to match the format above before saving to `data/`. This is the simplest approach.

**Option B — Edit `Characterization.py`** to match your column names. Find the `normalize_data()` method and update the `pd.read_csv()` call and the column references inside it:
```python
# Example: if your file uses 'V_pulse', 'dR_neg', 'R0_neg', 'dR_pos', 'R0_pos'
df = pd.read_csv(filepath)
df = df.rename(columns={
    'V_pulse':  'pulseAmplitude',
    'dR_neg':   'deltaRneg(measured at -80mV)',
    'R0_neg':   'RnegInitial',
    'dR_pos':   'deltaRpos(measured at +80mV)',
    'R0_pos':   'RposInitial',
})
```

#### Data normalisation

The `normalize_data()` method in `Characterization.py` preprocesses your raw file and saves a normalised copy as `data/YOUR_FILE_normalized.csv`. This step runs automatically on first use. If you modify your raw data, set `force_recompute == True` or delete the `_normalized.csv` file to force re-normalisation.

---

### 5.2 Choosing a Synapse Model

Open **`config.py`** and set `SYNAPSE_MODEL` to one of the two options:

```python
# config.py

SYNAPSE_MODEL = "Ferroelectric_Tanh"   # Use tanh resistance-envelope model (default)
# SYNAPSE_MODEL = "Ferroelectric"       # Use exponential switching model
```

**Which model should I choose?**

- Use `Ferroelectric_Tanh` (default). This model fits the upper and lower resistance envelopes using tanh curves, matching the hysteretic R–V switching behaviour of FTJ/FeFET-type devices.
- Use `Ferroelectric` if you want to fit an exponential-window VDSP model and use the co-design algorithm (§5.6).

---

### 5.3 Running Device Characterisation

Run the characterisation script to fit your data and visualise the model:

```bash
python Characterization.py
```

This will:
- Load your normalised device data.
- Fit the model parameters using least-squares curve fitting (`lmfit` / `scipy.optimize.curve_fit`).
- Save the fitted parameters to `data/params_SYNAPSE_MODEL.csv`.
- Display and save characterisation figures to the `figures/` directory.

> **Important:** The variable `save_path` on line 12 of `Characterization.py` is currently set to an absolute Windows path from the author's machine. Change it to a relative path before running:
> ```python
> save_path = "figures"   # ← change to this
> ```

**Fit result:**  
The script prints the fitted parameter values to the terminal and saves them as `data/params_Ferroelectric.csv` or `data/params_Ferroelectric_Tanh.csv`. These cached parameters are loaded automatically on subsequent runs — we recommend you set `force_recompute == True` in any case to force re-fitting.

#### Model parameters explained

**Tanh model (`Ferroelectric_Tanh`)** — output parameters:

| Parameter | Physical meaning |
|---|---|
| `r_min` | Minimum (on-state) resistance (Ω) — lower bound of the LTP envelope |
| `r_max` | Maximum (off-state) resistance (Ω) — upper bound of the LTD envelope |
| `v0_up` | Voltage scale of the upper (LTD) tanh envelope |
| `voff_up` | Voltage offset of the upper envelope |
| `v0_low` | Voltage scale of the lower (LTP) tanh envelope |
| `voff_low` | Voltage offset of the lower envelope |

**Exponential model (`Ferroelectric`)** — output parameters:

| Parameter | Physical meaning |
|---|---|
| `gamma_p` | LTP window exponent (controls saturation near W=1) |
| `gamma_d` | LTD window exponent (controls saturation near W=0) |
| `theta_p` | LTP switching threshold voltage |
| `theta_d` | LTD switching threshold voltage |
| `alpha_p` | LTP switching rate |
| `alpha_d` | LTD switching rate |

---

Note: You do not have to run this before run Training.py. This is just to show the visualized results of characterization, it will be automatically executed when running Training.py.

### 5.4 Configuring Device Variation Parameters

Variation coefficients are configured in `Characterization.py` inside the `MODEL_CONFIGS` dictionary (starting around line 18). All coefficients are **relative standard deviations** (e.g., `0.05` means 5% noise on the parameter value).

#### D2D (device-to-device) variation

```python
"device_to_device_variation_coefficient": {
    "gamma_p": 0.0,   # set to e.g. 0.10 for 10% D2D spread on gamma_p
    "gamma_d": 0.0,
    "theta_p": 0.0,
    "theta_d": 0.0,
    "alpha_p": 0.0,
    "alpha_d": 0.0
},
```

For `Ferroelectric_Tanh`, the keys are `r_min`, `r_max`, `v0_up`, `voff_up`, `v0_low`, `voff_low`.

D2D variation is injected at **initialisation time**: each synapse in the network receives an independently perturbed version of the base parameters. A synapse whose perturbed parameters violate physical constraints (e.g., `gamma_p <= 1`, `theta_p >= 0`) is marked as **defective** — its weight is frozen at 0 and does not participate in learning.

#### Defect Control

The criteria to define the defects could be changed if in need. This has to be done in two places: 1. Layer.py: modify `self.defect_mask`; 2. Synapse_Models: modify  `self.defect_mask` in the corresponding model class.

#### C2C (cycle-to-cycle) variation

```python
"cycle_to_cycle_variation_coefficient_multiplicative": 0.0,  # relative noise on delta_W
"cycle_to_cycle_variation_coefficient_additive": 0.0,        # absolute noise on delta_W
```

C2C variation adds Gaussian noise to each weight update at every VDSP step. Set one or both coefficients to a non-zero value (e.g., `0.05`) to simulate stochastic write noise.

---

### 5.5 Training the CSNN with Your Device

After characterisation and variation configuration, run:

```bash
python Training.py
```

Key parameters you may want to tune are in **`Training.py`**:

#### Dataset selection (lines ~27–30)

```python
# Switch between datasets:
full_dataset = datasets.FashionMNIST(...)   # 10-class clothing (default)
# full_dataset = datasets.MNIST(...)         # 10-class handwritten digits
```

#### Main training call (bottom of file, `__main__` block)

```python
main(
    train_csnn=True,          # True = retrain CSNN; False = load saved checkpoint
    sfp=1.138,                # LTP scaling factor (see §5.6 for how to set this)
    sfd=1.9,                  # LTD scaling factor — EDIT THIS for your device
    convergence_rate=0.14,    # Stop VDSP when weight polarisation falls below this
    v=1.02,                   # Reference potential v_ref
    train_svm=True,           # True = retrain SVM; False = load saved checkpoint
    is_feature_extraction=True
)
```

| Parameter | Where | What it does |
|---|---|---|
| `sfd` | `Training.py`, `main()` | LTD write-voltage scaling; the most important parameter to tune for a new device. Start with `sfd = 1.0` and increase until weights converge at a reasonable speed. |
| `sfp` | `Training.py`, `main()` | LTP write-voltage scaling; best determined by the co-design algorithm (§5.6). |
| `convergence_rate` | `Training.py`, `Train_csnn()` | VDSP convergence threshold. Lower values mean more training. Typical range: 0.08–0.20. |
| `v` | `Training.py`, `main()` | Reference membrane potential v_ref. Affects the amplitude of write pulses. Default: `1.02`. |
| `TIMESTEPS` | `config.py` | Number of simulation timesteps per image. Default: `20`. |

Note: The criteria of "convergence" could be changed if in need. This is done by line 197-215 in Training.py.

#### Saving and loading checkpoints

After training, the CSNN weights are saved to:
```
snn_full_model_epoch_1.pth          (root directory)
checkpoints_CSNN/snn_full_model_epoch_1.pth
```

The SVM model is saved to:
```
checkpoints_SVM/SVM_weight.pth
```

To skip retraining and only evaluate accuracy:
```python
main(
    train_csnn=False,
    train_svm=False,
    is_feature_extraction=False
)
```
Note: if you change any network parameters including decive data, retraining must be done. We recommend always retrain.
---

### 5.6 Co-Design Algorithm (Exponential Model Only)

For the `Ferroelectric` (Exponential) model, the optimal `sfp` value is found automatically by the **Iterative co-design algorithm** in `Solver.py`. This algorithm iteratively adjusts `sfp` to achieve a target asymmetry ratio between LTP and LTD strengths.

To run the co-design algorithm, set `SYNAPSE_MODEL = "Ferroelectric"` in `config.py`, then run:

```bash
python Training.py
```

The relevant parameters in the `__main__` block of `Training.py` are:

```python
target_beta = 1.05    # Target LTP/LTD asymmetry ratio. Tune if in need. We recommend a value slightly higher than 1.0.
sfd = 1.9             # Fixed LTD scale. Set this first based on your device and dataset to get a reasonable convergence speed.
v_ref = 1.0           # Reference potential. We recommend always set it to 1.0 with an epsilon to avoid singularity. 1.02 is used by Author.
initialGuess = 1.03   # Initial sfp estimate. Better guess is prefered for less iterative rounds but not necessary.
w_mean = 0.21         # Expected converged weight mean. Set to None for auto-detection. We recommend you to run only once and take down the final value and use it. 0.21 is a value verified by Author for Fashion-MNIST and MNIST.
EM_Round = 5          # Number of iterations (4 or 5 is usually sufficient).
convergence_rate = 0.14
```

The algorithm outputs a convergence plot (`EM Algorithm.png`) and automatically uses the final `sfp` value for the subsequent training runs.

> **For the `Ferroelectric_Tanh` model**, co-design is not required. No need to do anything but just run.

---

### 5.7 Evaluating D2D and C2C Robustness

After configuring variation coefficients (§5.4) and running `Training.py` at multiple variation strengths, use the plotting scripts to visualise the impact:

```bash
# D2D variation figures:
python D2D_ploting.py

# C2C variation figures:
python C2C_plotting.py
```

These scripts contain hard-coded accuracy dictionaries from the paper's experiments. To plot your own results, replace the `data` dictionary at the top of each script with your measured accuracies at each variation coefficient level.

---

## 6. Parameter Reference

### `config.py` — Global configuration

| Variable | Default | Description |
|---|---|---|
| `SYNAPSE_MODEL` | `"Ferroelectric_Tanh"` | Active synapse model. Options: `"Ferroelectric"`, `"Ferroelectric_Tanh"` |
| `TIMESTEPS` | `20` | Simulation timesteps per input image | This is a parameter dependent on the dataset. For Fashion-MNIST we set it to 20 and MNIST 15.
| `device` | `'cpu'` | Computation device. Set to `"cuda"` for GPU |

### `Characterization.py` — Device model config

| Location | Variable | Description |
|---|---|---|
| Line 12 | `save_path` | Output directory for characterisation figures. **Change to `"figures"` before running.** |
| Line 16 | `RAW_DEVICE_DATA_PATH` | Path to your raw device measurement file |
| `MODEL_CONFIGS[...]['variations']` | `device_to_device_variation_coefficient` | Per-parameter D2D noise coefficient (0 = ideal) |
| `MODEL_CONFIGS[...]['variations']` | `cycle_to_cycle_variation_coefficient_multiplicative` | C2C multiplicative noise coefficient |
| `MODEL_CONFIGS[...]['variations']` | `cycle_to_cycle_variation_coefficient_additive` | C2C additive noise coefficient |

### `Model.py` — Network architecture

| Parameter | Default | Description |
|---|---|---|
| `conv1` output channels | `128` | Number of convolutional filters |
| `kernel_size` | `7` | Convolution kernel size |
| `n_winners` | `7` | WTA lateral inhibition radius |
| `input_shape` | `(1,1,28,28)` | Input shape. **Must be updated if using non-28×28 images.** | Specifically, for Grayscale image, set channels to 1 when creating the instance and 3 for RGB.

### `Training.py` — Training hyperparameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `sfd` | `main()` argument | `1.9` | LTD voltage scaling factor | 1.9 for Fashion-MNIST and 1.3 for MNIST.
| `sfp` | `main()` argument | `1.138` | LTP voltage scaling factor | Do not modify values here, it will be automatically replaced by the co-design algorithm.
| `convergence_rate` | `main()` / `Train_csnn()` | `0.14` | VDSP stop criterion threshold | Be careful to change the criteria because it relies on the understanding of the system dynamics.
| `v` | `main()` argument | `1.02` | Membrane reference potential v_ref | 
| `VSDP_EPOCHS` | Line ~38 | `1` | Number of full dataset passes for VDSP |
| SVM training samples | `fit_svm` call | `60000` | Number of training samples for SVM |

---

## 7. Workflow Diagram

```
Your device data (*.dat)
        │
        ▼
[Characterization.py]
  curve_fit → data/params_*.csv
        │
        ▼
[config.py]
  SYNAPSE_MODEL = "Ferroelectric" or "Ferroelectric_Tanh"
  + variation coefficients in MODEL_CONFIGS
        │
        ▼
[Training.py]
  STAGE 1: VDSP training (Layers.py + Synapse_Models.py)
           ↕ weight checkpoints (checkpoints_CSNN/)
  STAGE 2: Feature extraction + SVM training
           ↕ SVM checkpoints (checkpoints_SVM/)
  STAGE 3: Test accuracy printed to terminal
        │
        ▼
[D2D_ploting.py / C2C_plotting.py]
  Variation robustness plots → figures/
```
All you have to do is to replace the device data (*.dat), set variation strength and run Training.py.
---

## 8. Troubleshooting

**`FileNotFoundError: data/ABS_03_summary.dat`**  
→ Your device data file is missing or `RAW_DEVICE_DATA_PATH` in `Characterization.py` line 16 is wrong.

**`KeyError` or `ValueError` when loading your data file**  
→ Your file's column names do not match the expected names. See §5.1 for the required column names and how to rename them in `Characterization.py → normalize_data()`.

**`FileNotFoundError` when saving figures**  
→ `save_path` in `Characterization.py` line 12 still points to the author's absolute Windows path. Change it to `"figures"`.

**`FileNotFoundError: checkpoints_CSNN/snn_full_model_epoch_1.pth`**  
→ You called `main(train_csnn=False)` but no checkpoint exists yet. Run with `train_csnn=True` first.

**CSNN weights do not converge (stuck near 0 or 1)**  
→ `sfd` or `sfp` is mismatched with your device's switching voltages. Try increasing `sfd` and never below  `1.0` and check that your characterisation fit is reasonable. For the Exponential model, run the co-design algorithm.

**Very low accuracy (near random ~10%)**  
→ The network may have collapsed. If it happens under very strong variations, it is the expected phenomenon.

**CUDA errors / want to use GPU**  
→ The code forces CPU by default. To enable GPU, change `device = 'cpu'` to `device = "cuda"` in `config.py`.

**Training is very slow**  
→ The simulation processes one image at a time. The reference speed is: one run by half an hour. Also setting `tau == 1.0` in model.py may increase the speed and gives very close results with `tau == 0.99`.

---

## 9. Expected Results

| Dataset | Model | Variation | Accuracy |
|---|---|---|---|
| Fashion-MNIST | `Ferroelectric_Tanh` | None (c=0) | ~88.3% |
| Fashion-MNIST | `Ferroelectric` | None (c=0) | ~88.5% |
| Fashion-MNIST | `Ferroelectric_Tanh` | D2D c=0.25 | ~87.5% |
| Fashion-MNIST | `Ferroelectric` | D2D c=0.30 | ~87.7% |
| Fashion-MNIST | `Ferroelectric_Tanh` | D2D c≥0.30 | Bifurcation (some seeds ~55% or ~10%) |
| Fashion-MNIST | `Ferroelectric` | D2D c=0.50 | ~87% (monotonic degradation) |

The `Ferroelectric` (Exponential) model is significantly **more robust** to D2D variation than the `Ferroelectric_Tanh` model, maintaining functional accuracy even at large variation strengths. The Tanh model exhibits a bifurcation transition near c=0.30 where some random seeds cause catastrophic accuracy collapse.
