import torch
import matplotlib.pyplot as plt
import joblib
from torchvision import datasets, transforms
import random

from Model import CSNN_Layerwise


def visualize_weights(model, save_name="weights_vis.png"):

    weights = model.conv1.weight.data.cpu().numpy()


    num_filters = weights.shape[0]  # 70
    kernel_size = weights.shape[2]  # 7


    grid_cols = 10
    grid_rows = (num_filters // grid_cols) + 1

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 8))
    fig.suptitle(f"Learned SNN Weights (Mean: {weights.mean():.3f})", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:

            w = weights[i, 0]

            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)

            ax.imshow(w, cmap='gray', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)
    print(f">>> weights visualized and saved to {save_name}。")


def predict_and_show(model, dataset, device, num_samples=10):
    model.is_svm_trained = True

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

    indices = random.sample(range(len(dataset)), num_samples)

    print(f"\n>>> predictions ({num_samples} pages)...")

    for i, idx in enumerate(indices):
        img, label = dataset[idx]

        img_tensor = img.to(device)
        try:
            pred_label = model.predict(img_tensor)
        except Exception as e:
            print(f"prediction Error: {e}")
            pred_label = "?"

        ax = axes[i]

        ax.imshow(img.squeeze().numpy(), cmap='gray')

        color = 'green' if pred_label == label else 'red'
        ax.set_title(f"Pred: {pred_label}\nTrue: {label}", color=color, fontweight='bold')
        ax.axis('off')

    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> current device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = CSNN_Layerwise(device=device)


    checkpoint_path = "checkpoints_CSNN/snn_weight_epoch_30.pth"

    try:
        print(f">>> loading weights: {checkpoint_path} ...")
        weights = torch.load(checkpoint_path, map_location=device)

        model.conv1.weight.data = weights
        print(">>> SNN weights loaded successfully！")
    except FileNotFoundError:
        print(f" Error! {checkpoint_path} not found")
        return

    visualize_weights(model)

    path = f"checkpoints_SVM/SVM_weight.pth"
    model.svm_classifier = joblib.load(path)
    model.is_svm_trained = True

    predict_and_show(model, test_dataset, device)

main()