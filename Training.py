import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import joblib
from Model import CSNN_Layerwise
from utils import DoGTransform
import matplotlib.pyplot as plt
from Solver import CoDesignSolver
from Layers import Timesteps


#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
print(f"current device: {device}")

SYNAPSE_MODEL = "Ferroelectric"

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
full_dataset = torch.utils.data.Subset(full_dataset, torch.randperm(len(full_dataset))[:len(full_dataset)])

##Please edit the transform of preprocessing input images.##

transform = transforms.Compose([
    #DoGTransform(device=device ,kernel_size=7, sigma1=1.0, sigma2=2.0)
])

##End of editing. DO REMEMBER TO EDIT THE INPUT SIZE OF DATASET IN MODEL.py ACCORDINGLY!!!##
train_cnt = 0
VSDP_EPOCHS = 1
samples_per_epoch_pcn = 6000
##Define samples_per_epoch_pcn for PCN training, and pcn_fit method of Model will automatically adjust to make every sample get used.##
##Feeding the whole training dataset to PCN training is recommended to ensure the model can see all training samples.##
def main(train_csnn,train_pcn,v=1.0,sfp=1.138,pcn_epochs=50,convergence_rate=0.1,is_pcn=True,train_svm=False,is_svm=False,is_feature_extraction=False):
##is_svm: True means use SVM as the readout layer.
##is_pcn: True means use PCN as the readout layer.(default)
##train_svm: True means train SVM classifier, False means load weights for SVM classifier.
##train_pcn: True means train PCN classifier, False means load weights for PCN classifier.(input by user)
##train_csnn: True means train CSNN, False means load weights for CSNN.(input by user)
##is_feature_extraction: True means extract features using current CSNN, False means load features extracted from the last CSNN.(input by user)
##for small size input images, the device is strongly recommended to be cpu.

    if train_csnn:
        is_feature_extraction=True #if you retrained CSNN, you must extract features using current CSNN!!!


    os.makedirs("checkpoints_CSNN", exist_ok=True)
    os.makedirs("checkpoints_SVM", exist_ok=True)
    os.makedirs("checkpoints_PCN", exist_ok=True)
    os.makedirs("extracted_feature", exist_ok=True)



##End of editing. DO REMEMBER TO EDIT THE INPUT SIZE OF DATASET IN MODEL.py ACCORDINGLY!!!##

    model = CSNN_Layerwise(device=device,synapse_model=SYNAPSE_MODEL,is_pcn=is_pcn,is_svm=is_svm,v=v,sfp=sfp)

    # ==========================================
    # STAGE 1: SNN unsupervised learning (VSDP)-
    # ==========================================
    if train_csnn:
        Train_csnn(model,convergence_rate=convergence_rate)
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

        svm_indices = torch.randperm(len(full_dataset))[:60000]
        svm_loader = DataLoader(Subset(full_dataset, svm_indices), batch_size=1, shuffle=False)

        X_list = []
        y_list = []
        if is_feature_extraction:
            print(">>> feature extraction...")
            for img, label in tqdm(svm_loader):
                img = img.to(device)
                img = transform(img)
                feat = model.feature_extractor(img)
                X_list.append(feat)
                y_list.append(label.item())
            print(">>> feature extraction complete.")
            X_tensor = torch.cat(X_list, dim=0) if X_list[0].dim() > 1 else torch.stack(X_list)
            y_tensor = torch.tensor(y_list, device=device)
            torch.save((X_tensor, y_tensor), "extracted_feature/extracted_feature_SVM.pt")
        else:
            (X_tensor, y_tensor) = torch.load("extracted_feature/extracted_feature_SVM.pt", map_location=device)

        print(">>> SVM training...")
        model.fit_svm(X_tensor, y_tensor,)
        path = f"checkpoints_SVM/SVM_weight.pth"
        joblib.dump(model.svm_classifier, path)
        print(">>> SVM training complete。")
    elif is_svm:
        path = f"checkpoints_SVM/SVM_weight.pth"
        model.svm_classifier = joblib.load(path)

    # ==========================================
    # STAGE 2: pcn Training
    # ==========================================
    if is_pcn & train_pcn:
        print("\n" + "=" * 50)
        print("STAGE 2: freeze SNN，use SNN as feature extractor for PCN")
        print("=" * 50)
        indices = torch.randperm(len(full_dataset))[:len(full_dataset)]
        loader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=True)
        print(">>> feature extraction...")

        X_list = []
        y_list = []
        if is_feature_extraction:
            for img, label in tqdm(loader):
                img = img.to(device)
                img = transform(img)
                feat = model.feature_extractor_pcn(img)
                X_list.append(feat)
                y_list.append(label.item())
            X_tensor = torch.cat(X_list, dim=0) if X_list[0].dim() > 1 else torch.stack(X_list)
            y_tensor = torch.tensor(y_list, device=device)
            torch.save((X_tensor, y_tensor), "extracted_feature/extracted_feature_PCN.pt")
        else:
            (X_tensor, y_tensor) = torch.load("extracted_feature/extracted_feature_PCN.pt", map_location=device)

        print(">>> PCN training...")
        print(f">>> training PCN (num_samples: {len(X_tensor)})...")

        X_sampled = X_tensor[indices]
        y_sampled = y_tensor[indices]

        print(">>> PCN training...")
        print(f">>> training PCN (num_samples: {len(X_list)})...")
        model.fit_pcn(X_sampled, y_sampled,pcn_epochs,samples_per_epoch_pcn)
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

    test_dataset = datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())
    test_times = 10
    accuracies = []

    for t in range(test_times):
        random_indices = torch.randperm(len(test_dataset))[:2500]
        test_subset = torch.utils.data.Subset(test_dataset, random_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=True)
        correct = 0
        total = 0
        for img, label in tqdm(test_loader, desc="Testing"):
            img = img.to(device)
            img = transform(img)
            pred = model.predict(img)
            if pred == label.item():
                correct += 1
            total += 1

        print(f"\n>>> Live Accuracy: {100 * correct / total:.2f}%")
        accuracies.append(100 * correct / total)
    accuracies=torch.tensor(accuracies)
    print(f"\n>>> Average Accuracy: {accuracies.mean():.2f}%")
    print(f"\n>>> std: {accuracies.std():.2f}%")
    return train_cnt,accuracies.mean(), accuracies.std()

def Train_csnn(model,convergence_rate=0.1):
    CONVERGENCE_RATE = convergence_rate
    weight_mean_list = []
    for epoch in range(VSDP_EPOCHS):

        train_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{VSDP_EPOCHS}")

        total_spikes = 0

        for train_cnt, (img, _) in enumerate(pbar):
            img = img.to(device)
            img = transform(img)

            _, out_spk = model.forward(img, is_training=True, lr=0.01)

            total_spikes += out_spk.sum().item()

            w = model.conv1.weight
            convergence = (w * (model.conv1.w_max - w)).mean().item()

            pbar.set_postfix({
                "Spikes": int(total_spikes),
                "Convg": f"{convergence:.4f}"
            })

            if convergence < CONVERGENCE_RATE:
                print(f"\n>>> meet the convergence rate ({CONVERGENCE_RATE})，CSNN training complete.")
                break

            weight_mean_list.append(w.mean().item())

        save_path = f"checkpoints_CSNN/snn_weight_epoch_{epoch + 1}.pth"
        torch.save(model.conv1.weight.data.cpu(), save_path)

        print(f">>> [Saved] epoch {epoch + 1} has been saved to: {save_path}")
        print(f">>> average spikes per image: {total_spikes / (train_cnt + 1):.2f}")
        # weight_mean = model.conv1.weight.mean()
        # weight_mean_list.append(weight_mean)

    print("\n>>> SNN training complete.")
    # x = range(len(weight_mean_list))
    # plt.figure()
    # plt.plot(x, weight_mean_list, linewidth=2)
    # plt.xlabel('iterations')
    # plt.ylabel('weight mean')
    # plt.grid(True, alpha=0.3)
    # plt.show()




if __name__ == "__main__":

    indices = torch.randperm(len(full_dataset))[:500]
    dataloader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=False)

    target_beta = 1.05
    v_ref = 1.0
    sfp = None
    sfd = 1.30
    initialGuess = 1.03
    w_mean = 0.21
    """
    if you already have the real converged weight mean, you can set w_mean to the real value.
    otherwise, set w_mean to None.
    """

    EM_Round=5
    current_sfp = initialGuess
    val_list = [current_sfp]

## Find the real converged weight mean.
    if w_mean is None:
        print(f"\nTrying to find the real converged weight mean (SFP: {current_sfp:.4f})...")
        temp_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd)
        Train_csnn(temp_model, convergence_rate=0.001)
        w_mean = temp_model.conv1.weight.mean().item()
    print(f">>> Real converged weight mean: {w_mean:.4f}")
##
    for i in range(EM_Round):
        print(f"\n--- EM Iteration Round {i} (Current SFP: {current_sfp:.4f}) ---")
        temp_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd)
        try:
            weight_path = f"checkpoints_CSNN/snn_weight_epoch_1.pth"
            weights = torch.load(weight_path, map_location=device)
            temp_model.conv1.weight.data = weights
            print(f">>> Loaded {weight_path}")
        except FileNotFoundError:
            print(">>> Warning: No weight file found, using initiated model weights.")

        solver = CoDesignSolver(timesteps=Timesteps)
        solver.extract_statistics(temp_model, dataloader, device)

        new_sfp = solver.solve_parameter(
            target_beta=target_beta,
            v_ref=v_ref,
            sfp=None,
            sfd=sfd,
            w_mean=w_mean
        )

        val_list.append(new_sfp)
        temp_model.sfp = new_sfp
        new_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd)
        Train_csnn(new_model, convergence_rate=0.1)
        current_sfp = (new_sfp+val_list[-2])/2

        if i > 0 and abs(val_list[-1] - val_list[-2]) < 0.001:
            print(">>> EM converged.")
            break

    plt.figure(figsize=(12, EM_Round))
    plt.plot(val_list, label="SFP")
    plt.xlabel("EM Round")
    plt.ylabel("SFP")
    plt.legend()
    plt.savefig('EM Algorithm.png', bbox_inches='tight', dpi=300)
    plt.show()

    if True:
        #main(False,True,is_svm=False,is_pcn=True,train_svm=False,is_feature_extraction=True)
        mean_list = []
        std_list = []
        num_samples_list = []
        convergence_rate_list=[0.158]
        pcn_training_epochs_list=[90,100,150,200]

        v_ref_list=[1.02]
        sfp_list=[
                  [1.1833, 1.1131, 1.0889, 1.0824, 1.062, 1.0465, 1.0377], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.0
                  #[1.0036,1.009,1.0144,1.018,1.027,1.036,1.09],#overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=0.8
                  #[1.0500, 1.0862, 1.1000, 1.1150, 1.1300, 1.1500, 1.1800, 1.2100, 1.2400, 1.2700, 1.3000, 1.3500, 1.4200, 1.5000], # overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.2
                  #[1.1500, 1.1863, 1.2187, 1.2400, 1.2800, 1.3500, 1.4385, 1.5500, 1.6578, 1.8000, 1.8770, 2.0500, 2.1500, 2.3155], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.5
                  #[1.1237, 1.3093, 1.4948, 1.6185, 1.9278, 2.2371, 2.8556], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.8
                  #[1.1477, 1.3693, 1.5908, 1.7385, 2.1078, 2.4771, 3.2156] #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=2.0
                  ]

        inout_list1=v_ref_list
        inout_list2=sfp_list
        for n,t in enumerate(v_ref_list):
            mean_list = []
            std_list = []
            num_samples_list = []
            for i in inout_list2[n]:
                train_cnt,mean,std=main(True,False,convergence_rate=0.1,v=t,sfp=i,is_svm=True,is_pcn=False,train_svm=True,is_feature_extraction=False)
                mean_list.append(mean.item())
                std_list.append(std.item())
                num_samples_list.append(train_cnt)
            print(f"input variable list:{inout_list2}\ntraining samples:{num_samples_list}\nmean:{mean_list}\nstd:{std_list}")