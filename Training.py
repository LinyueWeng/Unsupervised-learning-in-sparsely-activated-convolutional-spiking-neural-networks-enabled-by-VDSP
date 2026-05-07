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
from Model import TIMESTEPS
from Characterization import ModelCharac
from config import *

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n>>> [Seed Control] Random seed set to: {seed}")

print(f"current device: {device}")


# full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


full_dataset = torch.utils.data.Subset(full_dataset, torch.randperm(len(full_dataset))[:len(full_dataset)])

##Please edit the transform of preprocessing input images.##

transform = transforms.Compose([
    #DoGTransform(device=device ,kernel_size=7, sigma1=1.0, sigma2=2.0)
])

##End of editing. DO REMEMBER TO EDIT THE INPUT SIZE OF DATASET IN MODEL.py ACCORDINGLY!!!##
train_cnt = 0
VSDP_EPOCHS = 1

def main(train_csnn,v=1.0,sfp=1.138,sfd=1.9,convergence_rate=0.1,train_svm=False,is_feature_extraction=False):
    """
    The main function orchestrates the training and evaluation pipeline for a convolutional
    spiking neural network (CSNN) and its associated tasks. It allows the user to train the
    CSNN, extract features using the trained CSNN, train an SVM classifier and evaluate the
    entire pipeline's accuracy on a test set.

    :param train_csnn: A boolean indicating whether to train the CSNN model or load a saved one.
    :param v: A float representing the reference potential v_ref constant for the CSNN model,
        used in the layerwise training.
    :param sfp: A float parameter that sets a scaling factor for LTP
        in the Exponential Model. It is not defined here, but a value determined by the iterative algorithm
        defined in Solver.py.
    :param sfd: A float parameter that sets a scaling factor for LTD
        in the Exponential Model. It is defined here! Please edit the value according to your dataset and synaptic parameters.
    :param convergence_rate: A float representing the rate at which the CSNN model training
        converges.
    :param train_svm: A boolean indicating whether to train the Support Vector Machine (SVM)
        classifier.
    :param is_feature_extraction: A boolean that determines if feature extraction should be
        performed using the CSNN model.
    :return: A tuple containing the number of CSNN training samples used and the test accuracy
        as a percentage.
    """

    if train_csnn:
        is_feature_extraction=True #if you retrained CSNN, you must extract features using current CSNN!!!


    csnn_train_samples = 0

    if train_csnn:
        is_feature_extraction = True

    os.makedirs("checkpoints_CSNN", exist_ok=True)
    os.makedirs("checkpoints_SVM", exist_ok=True)
    os.makedirs("extracted_feature", exist_ok=True)

    model = CSNN_Layerwise(device=device, synapse_model=SYNAPSE_MODEL,
                            v=v, sfp=sfp,sfd=sfd)

    # ==========================================
    # STAGE 1: SNN unsupervised learning (VSDP)-
    # ==========================================
    if train_csnn:
        csnn_train_samples = Train_csnn(model, convergence_rate=convergence_rate)
    else:
        path = f"checkpoints_CSNN/snn_full_model_epoch_{VSDP_EPOCHS}.pth"
        checkpoint = torch.load(path, map_location=device)

    # ==========================================
    # STAGE 2: SVM Training
    # ==========================================
    if train_svm:
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
        model.fit_svm(X_tensor, y_tensor, )
        path = f"checkpoints_SVM/SVM_weight.pth"
        joblib.dump(model.svm_classifier, path)
        print(">>> SVM training complete。")
    else:
        path = f"checkpoints_SVM/SVM_weight.pth"
        model.svm_classifier = joblib.load(path)

    # ==========================================
    # STAGE 3: Test Accuracy (Test Set)
    # ==========================================
    print("\n" + "=" * 40)
    print("STAGE 3: Test Accuracy (Full Set)")
    print("=" * 40)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0

    for img, label in tqdm(test_loader, desc="Testing"):
        img = img.to(device)
        img = transform(img)
        pred = model.predict(img)
        if pred == label.item():
            correct += 1
        total += 1

    final_acc = 100 * correct / total
    print(f"\n>>> Test Accuracy: {final_acc:.2f}%")
    print(f">>> CSNN training samples used: {csnn_train_samples}")

    return csnn_train_samples, float(final_acc)


def Train_csnn(model, convergence_rate=0.1):
    CONVERGENCE_RATE = convergence_rate
    print(f"\n>>> Convergence rate: {CONVERGENCE_RATE}")

    dataset_obj = full_dataset.dataset if isinstance(full_dataset, Subset) else full_dataset
    is_mnist = isinstance(dataset_obj, datasets.MNIST)

    charac_model = ModelCharac(SYNAPSE_MODEL)
    base_params = charac_model()
    if SYNAPSE_MODEL == "Ferroelectric_Tanh":
        ratio = base_params['r_max'] / (base_params['r_min'])

    total_used_samples = 0

    for epoch in range(VSDP_EPOCHS):
        train_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{VSDP_EPOCHS}")
        total_spikes = 0
        epoch_used_samples = 0

        for train_cnt, (img, _) in enumerate(pbar, start=1):
            img = img.to(device)
            img = transform(img)

            _, out_spk = model.forward(img, is_training=True, lr=0.01)
            total_spikes += out_spk.sum().item()

            w1 = model.conv1.weight
            if SYNAPSE_MODEL == "Ferroelectric_Tanh":
                w_low = (1 / ratio)*0.1
            else:
                w_low = 0.0
            w_high = model.conv1.w_max.item() if torch.is_tensor(model.conv1.w_max) else model.conv1.w_max

            """
            here you define what the convergence rate is. you may change it to any criteria you like and set it to a reasonable value.
            """
            w_norm = (w1 - w_low) / (w_high - w_low)
            w_norm = torch.clamp(w_norm, 0.0, 1.0)
            #print(w1.mean().item())
            convg1 = (w_norm * (1.0 - w_norm)).mean().item()
            pbar.set_postfix({"Spikes": int(total_spikes), "C1": f"{convg1:.4f}"})

            epoch_used_samples = train_cnt
            if SYNAPSE_MODEL == "Ferroelectric_Tanh":
                stop_condition = convg1 < convergence_rate
            elif SYNAPSE_MODEL == "Ferroelectric":
                stop_condition = w_norm.mean() < 0.6
                #stop_condition = convg1 < convergence_rate

            if (stop_condition and train_cnt > 10) or (train_cnt > 300 and SYNAPSE_MODEL == "Ferroelectric_Tanh") or (train_cnt > 500 and SYNAPSE_MODEL == "Ferroelectric"):
                print(f"\n>>> CSNN training complete (Converged).")
                break

        total_used_samples += epoch_used_samples

        os.makedirs("checkpoints_CSNN", exist_ok=True)
        save_path = f"snn_full_model_epoch_{VSDP_EPOCHS}.pth"
        state_to_save = {'conv1': model.conv1.weight.data.cpu()}
        torch.save(state_to_save, save_path)
        print(f">>> [Saved] model saved to: {save_path}")

    print(f">>> Total CSNN training samples used: {total_used_samples}")
    return total_used_samples




if __name__ == "__main__":
    sfd = 1.0
    if SYNAPSE_MODEL == "Ferroelectric":
        """
        This part is the implementation of the do-design algorithm.
        """
        convergence_rate=0.14
        indices = torch.randperm(len(full_dataset))[:500]
        dataloader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=False)

        target_beta = 1.05
        v_ref = 1.0
        sfp = None
        sfd = 1.9
        initialGuess = 1.03
        w_mean = 0.21 #a value that is verified by the Author.
        """
        if you already know the real converged weight mean, you can set w_mean to the real value.
        otherwise, set w_mean to None.
        Note: this algorithm is not designed to find the strict boundaries of the optimal parameter space which is very difficult.
            This is just to automatically derive a sfp value that falls within the range of the optimal parameter space.
            So you do not need to find out a very precise value of the real converged weight mean but any value that is close to it.
        """

        EM_Round=5
        current_sfp = initialGuess
        val_list = [current_sfp]

    ## Find the real converged weight mean.
        if w_mean is None:
            """
            here you determine the final weight mean. It will be transmitted to the solver."""
            print(f"\nTrying to find the real converged weight mean (SFP: {current_sfp:.4f})...")
            temp_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd,synapse_model=SYNAPSE_MODEL)
            Train_csnn(temp_model, convergence_rate=convergence_rate*0.1)
            w_mean = temp_model.conv1.weight.mean().item()
            print(f">>> Real converged weight mean: {w_mean:.4f}")
    ##
        print(f"\n--- EM Iteration Round 0 (Current SFP: {current_sfp:.4f}) ---")
        temp_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd,synapse_model=SYNAPSE_MODEL)
        Train_csnn(temp_model, convergence_rate=convergence_rate)
        weight_path = f"checkpoints_CSNN/snn_weight_EM.pth"
        torch.save(temp_model.conv1.weight, weight_path)
        for i in range(EM_Round):
            print(f"\n--- EM Iteration Round {i} (Current SFP: {current_sfp:.4f}) ---")
            temp_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd,synapse_model=SYNAPSE_MODEL)
            try:
                weights = torch.load(weight_path, map_location=device)
                temp_model.conv1.weight.data = weights
                print(f">>> Loaded {weight_path}")
            except FileNotFoundError:
                print(">>> Warning: No weight file found, using initiated model weights.")

            solver = CoDesignSolver(timesteps=TIMESTEPS)
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
            new_model = CSNN_Layerwise(device=device, v=v_ref, sfp=current_sfp,sfd=sfd,synapse_model=SYNAPSE_MODEL)
            Train_csnn(new_model, convergence_rate=0.1)
            torch.save(new_model.conv1.weight, weight_path)
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


    #seeds = [42, 123, 2025, 7, 99,   4,16,2026,21,27,114,514,111,222,333,444,555,666,777,888]
    seeds = [42, 123, 2025, 7, 99,]
    v_ref_list = [1.02]
    sfp_list = [[1.0]]
    if SYNAPSE_MODEL == "Ferroelectric_Tanh":
        convergence_rate=0.14
    if SYNAPSE_MODEL == "Ferroelectric":
        sfp_list = [[val_list[-1]]]

    for n, t in enumerate(v_ref_list):
        for i in sfp_list[n]:
            experiment_results = []
            sample_results = []

            for s in seeds:
                set_seed(s)
                train_samples, acc = main(
                    train_csnn=True,
                    sfp=i,sfd=sfd,
                    convergence_rate=convergence_rate,
                    v=t,
                    train_svm=True,
                    is_feature_extraction=True
                )

                experiment_results.append(acc)
                sample_results.append(train_samples)

            acc_tensor = torch.tensor(experiment_results, dtype=torch.float32)
            sample_tensor = torch.tensor(sample_results, dtype=torch.float32)

            final_mean = acc_tensor.mean().item()
            final_std = acc_tensor.std(correction=0).item() if len(acc_tensor) > 1 else 0.0

            sample_mean = sample_tensor.mean().item()
            sample_std = sample_tensor.std(correction=0).item() if len(sample_tensor) > 1 else 0.0

            print(f"\n" + "!" * 30)
            print("FINAL RESULTS:")
            print(f"All Accuracies: {experiment_results}")
            print(f"Mean Accuracy: {final_mean:.2f}%")
            print(f"Accuracy Std: {final_std:.2f}%")
            print(f"All Training Samples: {sample_results}")
            print(f"Mean Training Samples: {sample_mean:.2f}")
            print(f"Training Samples Std: {sample_std:.2f}")
            print("!" * 30)

    if False:
        #main(False,True,is_svm=False,is_pcn=True,train_svm=False,is_feature_extraction=True)
        mean_list = []
        std_list = []
        num_samples_list = []
        convergence_rate_list=[[0.1,0.09,0.08]]
        pcn_training_epochs_list=[90,100,150,200]

        v_ref_list=[1.02]
        sfp_list=[
                    #[1.033]
                   [1.0]
                  #[1.1833, 1.1131, 1.0889, 1.0824, 1.062, 1.0465, 1.0377], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.0
                  #[1.0036,1.009,1.0144,1.018,1.027,1.036,1.09],#overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=0.8
                  #[1.0500, 1.0862, 1.1000, 1.1150, 1.1300, 1.1500, 1.1800, 1.2100, 1.2400, 1.2700, 1.3000, 1.3500, 1.4200, 1.5000], # overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.2
                  #[1.1500, 1.1863, 1.2187, 1.2400, 1.2800, 1.3500, 1.4385, 1.5500, 1.6578, 1.8000, 1.8770, 2.0500, 2.1500, 2.3155], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.5
                  #[1.1237, 1.3093, 1.4948, 1.6185, 1.9278, 2.2371, 2.8556], #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=1.8
                  #[1.1477, 1.3693, 1.5908, 1.7385, 2.1078, 2.4771, 3.2156] #overdrive: 20%,50%,80%,100%,150%,200%,300% v_ref=2.0
                  ]
        pcn_threshold_list=[
                            [60.0],
                            [1,5,10,15,20,30,40,50,60]
                    ]

        inout_list1=v_ref_list
        inout_list2=sfp_list
        for n,t in enumerate(v_ref_list):
            mean_list = []
            std_list = []
            num_samples_list = []
            for i in inout_list2[n]:
                train_cnt,mean,std=main()
                mean_list.append(mean.item())
                std_list.append(std.item())
                num_samples_list.append(train_cnt)
            print(f"input variable list:{inout_list2}\ntraining samples:{num_samples_list}\nmean:{mean_list}\nstd:{std_list}")