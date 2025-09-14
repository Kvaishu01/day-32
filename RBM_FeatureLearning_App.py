
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="RBM Feature Learning", layout="wide")
st.title("🔹 Day 32 — Restricted Boltzmann Machine (RBM) Feature Learning")

st.markdown("""
This app demonstrates how **Restricted Boltzmann Machines (RBMs)** can be used for feature learning 
on the handwritten digits dataset.  
RBMs are unsupervised neural networks that can learn compressed representations of data.
""")

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target
X = X / 16.0  # normalize 0-1

st.subheader("📂 Dataset Preview")
st.write("Digits dataset with shape:", X.shape)

# Show some digit images
fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray_r")
    ax.set_title(f"{y[i]}")
    ax.axis("off")
st.pyplot(fig)

# Sidebar settings
st.sidebar.header("⚙️ RBM Settings")
n_components = st.sidebar.slider("Number of hidden units", 25, 200, 64, step=25)
learning_rate = st.sidebar.selectbox("Learning rate", [0.01, 0.05, 0.1])
n_iter = st.sidebar.slider("Training epochs", 5, 30, 10)

# Build pipeline: RBM + Logistic Regression
rbm = BernoulliRBM(n_components=n_components, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
logistic = LogisticRegression(max_iter=1000)
classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

if st.button("Train RBM + Logistic Regression"):
    st.info("Training model, please wait...")
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    acc = accuracy_score(y, y_pred)
    st.success(f"✅ Training Accuracy: {acc:.3f}")

    st.text("Classification Report:")
    st.text(classification_report(y, y_pred))

    # Visualize RBM features
    st.subheader("🧩 Learned RBM Features (Hidden Units)")
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        if i < n_components:
            ax.imshow(rbm.components_[i].reshape(8, 8), cmap="gray_r")
        ax.axis("off")
    plt.suptitle("RBM Learned Features")
    st.pyplot(fig)
else:
    st.info("Adjust RBM settings and click **Train RBM + Logistic Regression** to start.")
