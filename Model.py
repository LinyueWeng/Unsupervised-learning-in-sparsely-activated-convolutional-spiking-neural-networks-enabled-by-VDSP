import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from Layers import CsnnLayer,SnnPooling
from Readout import ReadoutPCN
from sklearn.svm import LinearSVC
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys

#device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_local='cpu'
class CSNN_Layerwise:

    def __init__(self,synapse_model="Ferroelectric",device=device_local,is_pcn=True,is_svm=False,v=1.0,sfp=1.038,sfd=1.30):
        self.device = device
        self.timesteps = 15
        self.v_rest = 0
        self.v_thresh = 10
        self.v_reset = -1
        self.r_inhib = 3
        self.f_dep = 2
        self.lr = 0.01
        self.is_svm = is_svm
        self.is_pcn = is_pcn
        self.sfd=sfd
        self.synapse_model=synapse_model

        self.conv1 = CsnnLayer(self.input_shape(1),70,v=v,sfp=sfp,sfd=self.sfd,synapse_model=self.synapse_model,kernel_size=7,stride=1,padding=3,lr=self.lr,f_dep=self.f_dep,v_rest=self.v_rest,v_thresh=self.v_thresh,v_reset=self.v_reset,r_inhib=self.r_inhib,device=self.device)
        self.pool1 = SnnPooling(self.input_shape(70),kernel_size=3,stride=3,padding=0,v_thresh=1,timesteps=self.timesteps,device=self.device)
        if hasattr(self.conv1, 'weight'):
            self.conv1.weight = self.conv1.weight.to(self.device)

        if self.is_svm:
            self.svm_classifier = LinearSVC(C=0.005)
            self.is_svm_trained = False
        if self.is_pcn:
            self.pcn = ReadoutPCN(self.pool1.get_output_size(),10,device=self.device)
            if hasattr(self.pcn, 'weight'):
                self.pcn.weight = self.pcn.weight.to(self.device)



    def input_shape(self,channels):
        ##midify the size of each image here!##
        return 1,channels,28,28

    def forward(self,image_tensor,is_training=False,lr=0.01):
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        potential,spike=self.conv1.ttfs_inputlayer(image_tensor)
        potential,spike=self.conv1(potential,spike,is_training=is_training)
        #self.see_potential_evolve(potential)
        potential,spike=self.pool1(potential,spike)

        return potential,spike

    def feature_extractor(self,image_tensor):
        potential,spike=self.forward(image_tensor)
        feature=spike.sum(dim=0).flatten().to(self.device)
        return feature

    def fit_svm(self,train_data,train_label):
        print(f">>> training SVM (num_samples: {len(train_label)})...")
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()
        if isinstance(train_label, torch.Tensor):
            train_label = train_label.cpu().numpy()
        self.svm_classifier.fit(train_data,train_label)
        self.is_svm_trained=True

    def predict(self,image_tensor):
        if self.is_svm:
            feature=self.feature_extractor(image_tensor)
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            return self.svm_classifier.predict(feature.reshape(1,-1))[0]
        if self.is_pcn:
            feature=self.feature_extractor_pcn(image_tensor,t_max=1)
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            return self.pcn.predict(feature)
        else:
            print("Please choose either SVM or PCN!")
            return None

    def feature_extractor_pcn(self,image_tensor,t_max=1):
        ##feature is the normalized latency of the first spike over all timesteps##
        potential, spike = self.forward(image_tensor)
        timesteps, channels, height, width = spike.shape
        spike = spike.permute(1, 2, 3, 0)
        spike=spike.reshape(-1,timesteps).to(self.device)
        spike_mask=spike.max(dim=1)[0]>0
        first_spike=spike.argmax(dim=1).float()
        normalize_latency=first_spike/(timesteps-1) if timesteps>1 else first_spike
        feature=torch.where(spike_mask,normalize_latency,torch.ones_like(normalize_latency,device=self.device)*t_max)
        return feature#shape:(channels*height*width)

    def fit_pcn(self, X_tensor, y_tensor, num_epochs,num_samples_per_epoch):
        pcn_mean_list = []
        total_samples = len(X_tensor)
        chunks_per_pass = total_samples // num_samples_per_epoch

        print(f"\n>>> in total: {total_samples} samples，Every epoch has {num_samples_per_epoch} samples.")
        print(f">>> To traverse needs {chunks_per_pass} epochs. Currently epochs are set to be: {num_epochs}")

        all_indices = torch.randperm(total_samples, device=self.device)

        for epoch in range(num_epochs):

            current_chunk_idx = epoch % chunks_per_pass

            if current_chunk_idx == 0 and epoch > 0:
                print(f"\n>>> [Traversed，re shuffling and will start a new round...]")
                all_indices = torch.randperm(total_samples, device=self.device)

            start_idx = current_chunk_idx * num_samples_per_epoch
            end_idx = (current_chunk_idx + 1) * num_samples_per_epoch

            current_indices = all_indices[start_idx:end_idx]

            X_shuffled = X_tensor[current_indices]
            y_shuffled = y_tensor[current_indices]

            pbar = tqdm(range(len(X_shuffled)), desc=f"Epoch {epoch + 1}/{num_epochs}")
            X_batch = torch.as_tensor(X_shuffled, dtype=torch.float32).to(self.pcn.weight.device)
            y_batch = torch.as_tensor(y_shuffled).to(self.pcn.weight.device)
            for i in pbar:
                feat = X_batch[i]
                label = y_batch[i]

                self.pcn.s2_stdp_pcn_training(feat, label)
                pcn_mean_list.append(self.pcn.weight.mean().item())

            if (epoch + 1) % 100 == 0:
                self.pcn.pos_lr *= 0.5
                self.pcn.neg_lr *= 0.5
                print(f"\n>>> lr decay: pos_lr: {self.pcn.pos_lr:.5f}, neg_lr: {self.pcn.neg_lr:.5f}")
        print(">>> PCN training complete!")
        x=range(len(pcn_mean_list))
        plt.figure()
        plt.plot(x, pcn_mean_list, linewidth=2)
        plt.xlabel('iterations')
        plt.ylabel('weight mean')
        plt.grid(True, alpha=0.3)
        plt.show()

    def see_potential_evolve(self,potential,num_to_see=5):
            potential=potential.reshape(-1,self.timesteps)
            indices=torch.randperm(potential.shape[0],device=self.device)[:num_to_see]
            fig,axes=plt.subplots(num_to_see,1,figsize=(12,12))
            for i in range(num_to_see):
                axes[i].plot(potential[indices[i],:].cpu().numpy())
                axes[i].set_ylim(-2, 12)
                axes[i].set_title(f"Potential of neuron No. {indices[i]}")
                axes[i].grid(True,alpha=0.3)
            plt.tight_layout()
            plt.savefig('potential_evolve.png')
            plt.show()
            input("Potential image saved. Press Enter to exit...")
            sys.exit()
