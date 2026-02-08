import matplotlib.pyplot as plt


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation accuracy/loss from a Keras History object.
    """

    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    if acc is not None:
        plt.figure()
        plt.plot(acc, label="Train Accuracy")
        if val_acc is not None:
            plt.plot(val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title} - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    if loss is not None:
        plt.figure()
        plt.plot(loss, label="Train Loss")
        if val_loss is not None:
            plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title} - Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    