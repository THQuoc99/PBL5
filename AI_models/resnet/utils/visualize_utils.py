# utils/visualize_utils.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_emotions(images, labels, preds):
    """Trực quan hóa 5 ảnh test với nhãn thật và dự đoán."""
    emotions = ["Happy", "Negative", "Neutral", "Superise"]
    for i in range(min(5, len(images))):
        plt.imshow(images[i].reshape(48, 48), cmap="gray")
        plt.title(f"True: {emotions[labels[i]]}, Pred: {emotions[preds[i]]}")
        plt.show()