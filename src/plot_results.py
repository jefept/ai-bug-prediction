import matplotlib.pyplot as plt

models = ["Logistic Regression", "Random Forest", "SVM"]
scores = [0.72, 0.81, 0.78]

plt.bar(models, scores)

plt.ylabel("Accuracy")
plt.title("Model Comparison for Defect Prediction")

plt.show()
