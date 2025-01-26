---
marp: true
size: 16:9
# theme: uncover
headingDivider: 3
paginate: true
style: |
  section {
    text-align: left;
    background-color: white;
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* Aligns content to the top */
    font-size: 1.4em; /* Make the font smaller */
  }
---
# Training Ebru's 17-Dim Dynamic Scheduling Model

**Test instance:** 17 dimensions, 17 hours, 3060 time steps; 

**Plan from last meeting:**

- Implement the "ELU smoothing trick"
- Experiment 1: Test different reference policies
- Experiment 2: Test different reference policies with shape constraint (penalty for negative gradient)

**In this slide deck:**
- Experiment 1, 2 setup
- Experiment 1, 2 results  
- Appendix: Loss curves from all tests
- Next steps

## **Experiment 1 & 2 - Setup**

**Reference policy:** <span style="color: #0000FF;">**"minimal", "weighted-split", "even", "random"**</span>

**Hyperparameter settings:**

$$
\scriptsize
\begin{split}
\begin{array}{l|l}
\hline
\textbf{Hyperparameters} & \textbf{Values} \\
\hline
\text{Neural network architecture} & \text{MLP} \\
\text{Number of hidden layers} & 4 \\
\text{Number of nodes per layers} & \text{100} \\
\text{Activation function} & \text{LeakyReLU} \\
\text{Precision} & \text{float64} \\
\hline
\text{Optimizer} & \text{Adam} \\
\text{Batch size (training)} & \text{256} \\
\text{Batch size (validation)} & \text{512} \\
\text{Number of iterations} & \text{TBD (manual adjustment)} \\
\text{Learning rate schedule} & \text{Piecewise decay (manual)} \\
\text{Learning rates} & \text{Starts at $10^{-3}$, cut by $1/2$, minimum $10^{-5}$} \\
\hline
\text{ELU smoothing} & \color{blue} \text{Vanishing in 3000 steps} \\
\text{Shape constraint} & \color{blue} \textbf{Exp 1: None} \\
& \color{blue} \textbf{Exp 2: 1.0 on negative gradient} \\
\end{array}
\end{split}
$$

## **Experiment 1 & 2 - Result Summary**

$$
\scriptsize
\begin{split}
\begin{array}{l|l|l|l|l|l}
\hline
\textbf{Shape constr} & \textbf{Reference policy} & \textbf{End loss} & \textbf{Total steps} & \textbf{Average $V^{\mathrm{NN}}(0, X_0)$} & \textbf{NN policy cost} \\
\hline
\text{No} & \text{minimal} & \approx \color{blue} 7.35 & 10,000 & \approx 7.15 & \approx \color{blue} 1152.98 \\
\text{No} & \text{weighted split} & \approx 17.83 & 10,000 & \approx 8.18 & \approx 1183.70 \\
\text{No} & \text{even} & \approx 13.90 & 10,000 & \approx \color{blue} 41.03 & \approx 1155.89 \\
\text{No} & \text{random} & \approx 13.15 & 10,000 & \approx 38.53 & \approx 1155.15 \\
\hline
\text{Yes} & \text{minimal} & \approx \color{blue} 7.97 & 10,000 & \approx 54.70 & \approx \color{blue} 1091.07 \\
\text{Yes} & \text{weighted split} & \approx 18.52 & 10,000 & \approx \color{blue} 86.73 & \approx 1115.30 \\
\text{Yes} & \text{even} & \approx 13.90 & 10,000 & \approx 55.30 & \approx 1108.37 \\
\text{Yes} & \text{random} & \approx 12.30 & 10,000 & \approx 53.86 & \approx 1120.14 \\
\hline
\textbf{Benchmark} \\
\hline
\text{Ebru's} & & \approx 5 & \approx 30,000 & 80 \sim 120 & \approx 1050.0 \\
\text{$c\mu / \theta$ rule} & & & & & \approx 1040.0 \\
\end{array}
\end{split}
$$
Observations:
- The smoothing trick helps - it reduced cost from $\approx 1600$ to $\approx 1150$.
- Shape constraints then improve performance for all policies to $\approx 1100$.
- The "minimal" policy gives best loss and cost (Ebru's best was "weighted-split").

---
Averaged total cost by class:
$$
\scriptsize
\begin{split}
\begin{array}{l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l}
\hline
\textbf{Class} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} & \text{10} & \text{11} & \text{12} & \text{13} & \text{14} & \text{15} & \text{16} & \text{17} \\
\hline
\text{minimal} & 90 & 6 & 185 & 3 & 0 & 0 & 27 & 305 & 85 & 177 & 47 & 22 & 0 & 18 & 3 & 119 & 66 \\
\text{weighted split} & 65 & 2 & 254 & 1 & 0 & 0 & 39 & 305 & 80 & 162 & 47 & 19 & 0 & 18 & 3 & 119 & 68 \\
\text{even} & 81 & 17 & 176 & 45 & 0 & 0 & 53 & 268 & 82 & 200 & 31 & 21 & 2 & 23 & 6 & 108 & 45 \\
\text{random} & 73 & 15 & 189 & 38 & 0 & 0 & 49 & 271 & 82 & 196 & 31 & 21 & 1 & 23 & 4 & 117 & 45 \\
\hline
\text{minimal, with shape constraint} & 38 & 30 & 108 & 71 & 0 & 0 & 21 & 236 & 90 & 282 & 30 & 28 & 3 & 22 & 3 & 85 & 44 \\
\text{weighted split, with shape constraint} & 53 & 12 & 168 & 47 & 0 & 0 & 13 & 201 & 93 & 299 & 25 & 33 & 1 & 21 & 4 & 82 & 65 \\
\text{even, with shape constraint} & 47 & 26 & 138 & 62 & 0 & 0 & 27 & 218 & 90 & 288 & 28 & 30 & 3 & 22 & 3 & 82 & 47 \\
\text{random, with shape constraint} & 51 & 23 & 162 & 54 & 0 & 0 & 18 & 216 & 92 & 284 & 30 & 31 & 3 & 22 & 3 & 81 & 49 \\
\hline
c\mu / \theta \text{ rule} & 0 & 94 & 0 & 177 & 1 & 0 & 0 & 156 & 110 & 442 & 9 & 77 & 4 & 40 & 1 & 46 & 0 \\
\end{array}
\end{split}
$$
*The main difference is in how Class 3 and 4 are handled.*

## **All Loss Curves**

**Experiment 1 (smoothing only) - Policy "minimal"**
![](LR_SM3000_PLminimal_NP0.0/training_history.png)

---

**Experiment 1 (smoothing only) - Policy "weighted split"**
![](LR_SM3000_PLweighted_split_NP0.0/training_history.png)

---

**Experiment 1 (smoothing only) - Policy "even"**
![](LR_SM3000_PLeven_NP0.0/training_history.png)

---

**Experiment 1 (smoothing only) - Policy "random"**
![](LR_SM3000_PLrandom_NP0.0/training_history.png)

---

**Experiment 2 (smoothing and shape constraint) - Policy "minimal"**
![](LR_SM3000_PLminimal_NP1.0/training_history.png)

---

**Experiment 2 (smoothing and shape constraint) - Policy "weighted split"**
![](LR_SM3000_PLweighted_split_NP1.0/training_history.png)

---

**Experiment 2 (smoothing and shape constraint) - Policy "even"**
![](LR_SM3000_PLeven_NP1.0/training_history.png)

---

**Experiment 2 (smoothing and shape constraint) - Policy "random"**
![](LR_SM3000_PLrandom_NP1.0/training_history.png)

## **Next Steps**

- Discuss results with Ebru to find improvements
- Test more policy and penalty combinations 
- Try batch normalization