# Kaggle Project Report - Machine Learning (ModIA S8)

**Authors:** [Your Name 1] & [Your Name 2]  
**Date:** June 2026  
**Institution:** INP ENSEEIHT  

---

## 1. Learn how to use PyTorch
*Briefly describe your experience learning PyTorch. (e.g., "We familiarized ourselves with PyTorch tensors, autograd, and neural network modules (`nn.Module`) using the official PyTorch tutorials...").*

---

## 2. Train State-of-the-Art CNN Models

### Question 2.1: Image Pre-processing and Data Augmentation
*Explain how you pre-processed the TILDA textile texture images.*
*   **Resize:** Input images are resized to $64 \times 64$ (or $128 \times 128$) pixels to match standard CNN input dimensions.
*   **Normalization:** Normalized to speed up convergence.
*   **Data Augmentation:** Describe the augmentation techniques used during training (e.g., random horizontal flips, random rotations, random cropping) to prevent overfitting.

### Question 2.2: Implement CNN Architectures
*Describe the architecture of the models you implemented.*

| Model Name | Description / Number of Layers | Key Details (Kernel size, Pooling, Dropout) | Rationale for Choice |
| :--- | :--- | :--- | :--- |
| **LeNet-5** | 2 Conv layers, 2 MaxPool layers, 3 Fully Connected layers | Conv 5x5, MaxPool 2x2, ReLU activation | Classic baseline for digit and basic image recognition; simple and fast. |
| **AlexNet** | 5 Conv layers, 3 MaxPool layers, 3 Fully Connected layers | Conv 11x11, 5x5, 3x3; Dropout (0.5), AdaptiveAvgPool | Deeper baseline introducing Dropout and larger receptive fields for more complex patterns. |
| **ResNet-18** | 18 residual layers (blocks with skip connections) | Standard ResNet blocks, batch normalization, global avg pool | State-of-the-art deep residual architecture to prevent vanishing gradients and capture highly abstract textures. |

*Detailed specifications:*
*   **LeNet:** Detailed in [lenet.py](file:///Users/arkanyota/ProjetFilRouge2/src/models/lenet.py).
*   **AlexNet:** Detailed in [alexnet.py](file:///Users/arkanyota/ProjetFilRouge2/src/models/alexnet.py).
*   **ResNet-18:** Detailed in [resnet.py](file:///Users/arkanyota/ProjetFilRouge2/src/models/resnet.py).

### Question 2.3: Training Setup & Performance
*Describe the training parameters (optimizer, learning rate, weight decay, epochs) and report training times.*

*   **Optimizer:** Mini-batch SGD (Stochastic Gradient Descent) with momentum = 0.9.
*   **Learning Rate:** 0.01 (or tuned).
*   **Weight Decay:** 1e-4 (L2 regularization to avoid overfitting).

#### 🛠️ Commands to Run to Get these Results:
```bash
# Run LeNet-5 (15 epochs)
./venv/bin/python3 main.py --model lenet --epochs 15 --data_dir data/tilda

# Run AlexNet (15 epochs)
./venv/bin/python3 main.py --model alexnet --epochs 15 --data_dir data/tilda

# Run ResNet-18 (15 epochs)
./venv/bin/python3 main.py --model resnet --epochs 15 --data_dir data/tilda
```

*Fill the training performance table below after running the commands:*

| Model | Epochs | Batch Size | Total Training Time (seconds) | Best Validation Accuracy (%) | Test Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** | 15 | 16 | [Fill] | [Fill] | [Fill] |
| **AlexNet** | 15 | 16 | [Fill] | [Fill] | [Fill] |
| **ResNet-18** | 15 | 16 | [Fill] | [Fill] | [Fill] |

### Question 2.4: Validation & Tuning
*Analyze overfitting and describe parameter tuning.*
*   Include the saved training/validation curves (e.g. `lenet_multiclass_learning_curves.png`, `alexnet_multiclass_learning_curves.png`, `resnet_multiclass_learning_curves.png`) to show training loss vs validation loss.
*   *Interpretation:* Analyze if the gap between train loss and validation loss grows (indicating overfitting) and how weight decay / dropout mitigated it.

---

## 3. Machine Learning and AI Safety

### 3.1. Bias Introduction in TILDA

#### Question 3.1: Real-Life Bias Situation and Consequences
*Describe a real-world scenario where machine learning models can be biased.*
*   **Example:** Facial recognition systems trained on imbalanced racial datasets performing worse on minority groups, or loan-approval models biased against specific demographics due to historical training data bias.
*   **Consequences:** Systematic discrimination, unfair treatment, and reinforcement of social inequalities.

#### Question 3.2: Biased Dataset Construction
*Explain the mathematical approach implemented in [dataset.py](file:///Users/arkanyota/ProjetFilRouge2/src/dataset.py):*
*   We draw a Bernoulli variable $S \in \{0, 1\}$ with probability $p_0$ for class 0 (e.g., textile label 0) and $p_1$ for class 1 (e.g., textile label 1).
*   We append a channel $\epsilon$ representing random noise (Gaussian $\mathcal{N}(0, I)$) when $S = 1$, and zeros when $S = 0$.
*   This creates a spurious correlation between the noise channel and the label, simulating a dataset bias.

---

### 3.2. Model Bias Evaluation

#### 🛠️ Commands to Run to Get these Results:
```bash
# 1. Train Model 1 (Unbiased Dataset: p0 = 0.5, p1 = 0.5)
./venv/bin/python3 main.py --model lenet --epochs 10 --data_dir data/tilda --binary --p0 0.5 --p1 0.5

# 2. Train Model 2 (Biased Dataset: p0 = 0.0, p1 = 1.0)
./venv/bin/python3 main.py --model lenet --epochs 10 --data_dir data/tilda --binary --p0 0.0 --p1 1.0
```

#### Question 3.3: Experimental Study of Bias (DI Metric)
*Analyze the Disparate Impact (DI) of both models on test data.*
$$\text{DI} = \frac{\mathbb{P}(\widehat{y}(X) = 1 | S = 0)}{\mathbb{P}(\widehat{y}(X) = 1 | S = 1)}$$
*   **Model 1 (Unbiased Train):** [Fill DI metric]
*   **Model 2 (Biased Train):** [Fill DI metric]

#### Question 3.4: Test Accuracy Comparison
*Compare the classification performance of both models on the test set.*
*   **Model 1 Test Accuracy:** [Fill]%
*   **Model 2 Test Accuracy:** [Fill]%

#### Question 3.5: Summary and Conclusions
*Summarize results in the table below:*

| Setting / Model | Training Bias ($p_0, p_1$) | Validation Accuracy (%) | Test Accuracy (%) | Disparate Impact (DI) on Test |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | $p_0 = 0.5$, $p_1 = 0.5$ | [Fill] | [Fill] | [Fill] |
| **Model 2** | $p_0 = 0.0$, $p_1 = 1.0$ | [Fill] | [Fill] | [Fill] |

*Conclusions:*
*   *Explain how the biased dataset affects Model 2.* (e.g., "Model 2 achieves high/low accuracy because it learns to look at the noise channel $\epsilon$ rather than actual image textures. This is evidenced by a DI metric far from 1, demonstrating high bias...").

---

### 3.3. Bias Study in the Literature (Bonus)

#### Question 3.6: "Est-ce qu'on peut faire confiance à mon modèle ?"
*Write a short essay (approx. 1 page) based on article [4] and CNIL reports [5].*
*   **Key Themes:**
    1. **EU AI Act compliance:** Requirements for high-risk AI systems (accuracy, robustness, cybersecurity, data quality/bias mitigation).
    2. **Trustworthiness:** Why model accuracy alone does not imply trust. How models can exploit shortcuts (like the noise $\epsilon$ in this project).
    3. **Auditing and Transparency:** Best practices from CNIL to audit model bias, guarantee fairness, and provide user explainability.

---

## References
*   [1] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.
*   [2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Communications of the ACM*, 60(6):84–90, 2017.
*   [3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. *CVPR*, 2016.
*   [4] Hal science article on AI safety: https://hal.science/hal-03253111
*   [5] CNIL guidelines on AI compliance: https://www.cnil.fr/fr/intelligence-artificielle/guide
