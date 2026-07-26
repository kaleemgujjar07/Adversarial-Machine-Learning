# 🧠 Adversarial Machine Learning Demo (CIFAR-10)

An interactive demo showing how a small, carefully crafted amount of noise — invisible to the human eye — can fool a deep learning image classifier into making a confidently wrong prediction.

**🔗 Live interactive demo:** [Try it here]([INSERT YOUR STREAMLIT LINK HERE])

---

## What this project shows

Deep learning models can reach very high accuracy on standard image benchmarks, yet still be fooled by inputs that look completely normal to a human. This project demonstrates that vulnerability directly: it takes a trained image classifier, applies a well-known adversarial attack to a real image, and shows the model's prediction flip from correct to wrong — even though the image looks unchanged to the eye.

## How the attack works (FGSM)

The **Fast Gradient Sign Method (FGSM)** is a classic "white-box" adversarial attack (meaning it assumes access to the model's internals). It works in three steps:

1. Calculate how the model's loss (its "wrongness") would change with respect to each pixel in the input image.
2. Take the *sign* of that gradient — i.e., just the direction (increase or decrease) that would push the loss up the fastest for each pixel.
3. Nudge every pixel slightly in that direction, scaled by a small amount called **epsilon**.

The result: a visually near-identical image that pushes the model just far enough across its internal decision boundary to misclassify it.

## Model & dataset

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Base model:** ResNet18, implemented in PyTorch
- **Attack library:** [Foolbox](https://foolbox.jonasrauber.de/) — a widely-used Python toolkit for adversarial attacks

## Tech stack

- **Deep learning:** PyTorch, Torchvision
- **Adversarial attacks:** Foolbox
- **Web app:** Streamlit
- **Model hosting:** Hugging Face Hub (keeps the GitHub repo lightweight — the model weights download automatically when the app runs)

## Using the demo

- Use the **Image Index** slider to pick a different CIFAR-10 test image if a particular one resists the attack — some images sit further from the model's decision boundary than others.
- Increase **Epsilon** (e.g., to 0.08) for a stronger, more reliable attack if a smaller value doesn't change the prediction.

## Running it locally

```bash
git clone https://github.com/kaleemgujjar07/adversarial-machine-learning.git
cd adversarial-machine-learning
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project structure

```
adversarial-machine-learning/
├── app.py                  # Streamlit app + attack logic
├── requirements.txt        # Python dependencies
└── README.md
```

Model weights (`cifar_resnet18.pth`) are hosted on Hugging Face Hub rather than committed to the repo, and are downloaded automatically the first time the app runs.


