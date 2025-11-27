import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_logs(csv_file):
    """
    Đọc dữ liệu từ file CSV và vẽ 4 đồ thị:
    - Train Loss / Epoch
    - Train Accuracy / Epoch
    - Validation Loss / Epoch
    - Validation Accuracy / Epoch
    """
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_file)

    # Kiểm tra dữ liệu
    if df.empty:
        print("CSV file is empty. No data to plot.")
        return

    epochs = df["Epoch"]
    train_loss = df["Train Loss"]
    train_accuracy = df["Train Accuracy"]
    val_loss = df["Val Loss"]
    val_accuracy = df["Val Accuracy"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Train Loss
    plt.subplot(2, 2, 1)
    sns.lineplot(x=epochs, y=train_loss, marker='o', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss per Epoch")
    plt.xticks(epochs)  # Hiển thị rõ từng giá trị của epoch

    # Train Accuracy
    plt.subplot(2, 2, 2)
    sns.lineplot(x=epochs, y=train_accuracy, marker='o', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Train Accuracy per Epoch")
    plt.xticks(epochs)  # Hiển thị rõ từng giá trị của epoch

    # Validation Loss
    plt.subplot(2, 2, 3)
    sns.lineplot(x=epochs, y=val_loss, marker='s', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss per Epoch")
    plt.xticks(epochs)  # Hiển thị rõ từng giá trị của epoch

    # Validation Accuracy
    plt.subplot(2, 2, 4)
    sns.lineplot(x=epochs, y=val_accuracy, marker='s', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy per Epoch")
    plt.xticks(epochs)  # Hiển thị rõ từng giá trị của epoch

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "data/logs/training_log.csv"  # Đường dẫn file CSV
    plot_training_logs(csv_path)
