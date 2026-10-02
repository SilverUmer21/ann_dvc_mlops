import json 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = tf.keras.models.load_model("models/model.h5")

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose = 1
)

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis = 1)

confusion_mat = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
disp.plot()
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed.")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Metrics saved to metrics.json")