# infer-hok

**infer-hok** is a project that implements a PyTorch-based surrogate system for approximating differential operators in the Local Anisotropic Basis Function Method (LABFM). This framework focuses on replacing high-order mesh-free differential operators with machine-learned surrogates.

The primary objective is to test different neural network models as surrogates and evaluate their accuracy through convergence analysis, providing insights into their effectiveness when replacing traditional numerical solvers.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Customization](#customization)

## Features

- **Neural Operator Surrogates:** Learn LABFM differential operators using neural networks.
- **Multiple Architectures:** Easily plug in different model types for comparison.
- **Convergence Analysis Tools:** Includes scripts to evaluate and plot convergence rates of surrogate approximations.
- **Object-Oriented Design:** Modular structure for easy customization and extension.
- **Torch-Based Implementation:** Leverages PyTorch for flexible training and inference workflows.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lucasstarepravo/infer-hok.git

 2. **Install Dependencies:**
 ```bash
 pip install -r requirements.txt
```

## Usage

The main execution script is `main.py`, which is designed to run inference and convergence evaluations with the selected surrogate model. Configure the key parameters directly in the script:

- **total_nodes_list:** Controls domain resolution (e.g., `[10, 20, 50, 100, 200, 400]`).
- **polynomial_list:** Should be `[2, 2, 2, 2, 2, 2]` for compatibility; must match the length of `total_nodes_list`.

To run the surrogate evaluation and convergence analysis:

```bash
python main.py
```
Plotting utilities such as `plot_convergence1`, `plot_convergence2`, and `plot_convergence3` are available for visualizing the results.

## Requirements

Please refer to `requirements.txt` for the full list of required Python packages.

## Customization

- **Model Architecture:** Switch or modify the model architecture being used in `models/` and the model in `functions/discrete_operator.py`.
- **Domain Resolution:** Modify `total_nodes_list` in `main.py` to test the surrogate at different mesh resolutions.
- **Convergence Evaluation:** Update or add new plotting scripts to analyze different aspects of model performance.
