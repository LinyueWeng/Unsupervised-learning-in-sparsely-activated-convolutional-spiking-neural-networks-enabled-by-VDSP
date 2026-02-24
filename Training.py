import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import joblib
from Model import CSNN_Layerwise
import os

def main(train_csnn,train_pcn,is_pcn=True,train_svm=False,is_svm=False):
##is_svm: True means use SVM as the readout layer.
##is_pcn: True means use PCN as the readout layer.(default)
##train_svm: True means train SVM classifier, False means load weights for SVM classifier.
##train_pcn: True means train PCN classifier, False means load weights for PCN classifier.(input by user)
##train_csnn: True means train CSNN, False means load weights for CSNN.(input by user)
##for small size input images, the device is strongly recommended to be cpu.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    print(f"current device: {device}")



    VSDP_EPOCHS = 99
    SAMPLES_PER_EPOCH = 500

    os.makedirs("checkpoints_CSNN", exist_ok=True)
    os.makedirs("checkpoints_SVM", exist_ok=True)
    os.makedirs("checkpoints_PCN", exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    model = CSNN_Layerwise(device,is_pcn=is_pcn,is_svm=is_svm)

    # ==========================================
    # STAGE 1: SNN unsupervised learning (VSDP)-
    # ==========================================
    if train_csnn:
        for epoch in range(VSDP_EPOCHS):

            indices = torch.randperm(len(full_dataset))[:SAMPLES_PER_EPOCH]
            subset_data = Subset(full_dataset, indices)
            #you must shuffle the dataset before training!(batch size=1)#
            train_loader = DataLoader(subset_data, batch_size=1, shuffle=True)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{VSDP_EPOCHS}")

            total_spikes = 0
            lr=0.01*(epoch+11)*(1/11)#convert lr within [0.01,0.1] with epoch within [0-99]

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

            print(f">>> [Saved] epoch {epoch + 1} has been saved to: {save_path}")
            print(f">>> average spikes per image: {total_spikes / SAMPLES_PER_EPOCH:.2f}")

        print("\n>>> SNN training complete.")
    elif not train_csnn:
        path = f"checkpoints_CSNN/snn_weight_epoch_{VSDP_EPOCHS}.pth"
        weights = torch.load(path, map_location=device)
        model.conv1.weight.data = weights

    # ==========================================
    # STAGE 2: SVM Training
    # ==========================================
    if is_svm & train_svm:
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
    elif is_svm:
        path = f"checkpoints_SVM/SVM_weight.pth"
        weights = torch.load(path, map_location=device)
        model.svm_classifier.weight= weights

    # ==========================================
    # STAGE 2: pcn Training
    # ==========================================
    if is_pcn & train_pcn:
        print("\n" + "=" * 50)
        print("STAGE 2: freeze SNN，use SNN as feature extractor for PCN")
        print("=" * 50)
        indices = torch.randperm(len(full_dataset))[:60000]
        loader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=True)
        print(">>> feature extraction...")

        X_list = []
        y_list = []
        for img, label in tqdm(loader):
            img = img.to(device)

            feat = model.feature_extractor_pcn(img)
            X_list.append(feat)
            y_list.append(label.item())

        print(">>> PCN training...")
        print(f">>> training PCN (num_samples: {len(X_list)})...")

        model.fit_pcn(X_list, y_list)
        path = f"checkpoints_PCN/PCN_weight.pth"
        torch.save(model.pcn.weight.data.cpu(), path)
        print(f">>> PCN weights mean: {model.pcn.weight.data.mean()}")
        print(">>> PCN training complete。")

    elif is_pcn:
        path = f"checkpoints_PCN/PCN_weight.pth"
        weights = torch.load(path, map_location=device)
        model.pcn.weight.data= weights




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

main(False,True,is_svm=False,is_pcn=True,train_svm=False)