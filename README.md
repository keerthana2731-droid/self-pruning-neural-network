 Self-Pruning Neural Network using PyTorch

 Problem Statement
This project implements a self-pruning neural network that dynamically removes unnecessary weights during training. The goal is to reduce model size and improve efficiency without significantly affecting accuracy.

---

 Approach

- Built a custom `PrunableLinear` layer instead of using standard layers
- Each weight is associated with a learnable **gate parameter**
- Gates are passed through a **sigmoid function (0 to 1)**
- Effective weight = Weight × Gate
- Used **L1 regularization** on gates to encourage sparsity

🔹 Loss Function
Total Loss = Classification Loss + λ × Sparsity Loss

---

Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.001  | 22.94%  |  0.00%   |
| 0.01   | 20.47%  | 0.00%    |
| 0.1    | 21.26%  | 0.00%    |

---

 Gate Distribution

The following graph shows how gate values are distributed after training:

![Gate Distribution](results.png)

---

 Key Insights

- Increasing λ increases sparsity (more weights are pruned)
- Higher sparsity can reduce model accuracy
- Shows a clear trade-off between efficiency and performance

---

 Technologies Used

- Python
- PyTorch
- Matplotlib

---

How to Run

 1. Install dependencies
```bash
pip install -r requirements.txt
