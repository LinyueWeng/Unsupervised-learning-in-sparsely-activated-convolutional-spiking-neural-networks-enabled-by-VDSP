import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import joblib
from Model import CSNN_Layerwise
import os

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    print(f"current device: {device}")



    VSDP_EPOCHS = 30
    SAMPLES_PER_EPOCH = 300

    os.makedirs("checkpoints_CSNN", exist_ok=True)
    os.makedirs("checkpoints_SVM", exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    model = CSNN_Layerwise(device)

    # ==========================================
    # STAGE 1: SNN unsupervised learning (VSDP)
    # ==========================================
    if True:
        for epoch in range(VSDP_EPOCHS):

            indices = torch.randperm(len(full_dataset))[:SAMPLES_PER_EPOCH]
            subset_data = Subset(full_dataset, indices)
            train_loader = DataLoader(subset_data, batch_size=1, shuffle=False)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{VSDP_EPOCHS}")

            total_spikes = 0
            lr=0.01*(epoch+1)/2

            for img, _ in pbar:
                img = img.to(device)

                _, out_spk = model.forward(img, is_training=True,lr=lr)

                total_spikes += out_spk.sum().item()

                pbar.set_postfix({
                    "Spikes": int(total_spikes),
                    "W_Mean": f"{model.conv1.weight.mean().item():.3f}"
                })

            save_path = f"checkpoints_CSNN/snn_weight_epoch_{epoch + 1}.pth"
            torch.save(model.conv1.weight.data.cpu(), save_path)

            print(f">>> [Saved] 第 {epoch + 1} 轮权重已保存至: {save_path}")
            print(f">>> average spikes per image: {total_spikes / SAMPLES_PER_EPOCH:.2f}")

        print("\n>>> SNN training complete.")

    # ==========================================
    # STAGE 2: SVM Training
    # ==========================================
    print("\n" + "=" * 50)
    print("STAGE 2: freeze SNN，use SNN as feature extractor for SVM")
    print("=" * 50)

    svm_indices = torch.randperm(len(full_dataset))[:50000]
    svm_loader = DataLoader(Subset(full_dataset, svm_indices), batch_size=1, shuffle=False)

    X_list = []
    y_list = []

    print(">>> feature extraction...")
    for img, label in tqdm(svm_loader):
        img = img.to(device)

        feat = model.feature_extractor(img)
        X_list.append(feat)
        y_list.append(label.item())

    print(">>> SVM training...")
    model.fit_svm(X_list, y_list)
    path = f"checkpoints_SVM/SVM_weight.pth"
    joblib.dump(model.svm_classifier, path)
    print(">>> SVM training complete。")

    # ==========================================
    # STAGE 3: Test Accuracy (Test Set)
    # ==========================================
    print("\n" + "=" * 40)
    print("STAGE 3: Test Accuracy (Test Set)")
    print("=" * 40)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    correct = 0
    total = 0

    for img, label in tqdm(test_loader, desc="Testing"):

        pred = model.predict(img)
        if pred == label.item():
            correct += 1
        total += 1

    print(f"\n>>> Accuracy: {100 * correct / total:.2f}%")


main()