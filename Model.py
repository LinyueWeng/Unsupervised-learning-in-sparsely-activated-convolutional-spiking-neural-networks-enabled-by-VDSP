import torch
from Layers import CsnnLayer,SnnPooling
from Readout import ReadoutPCN
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tqdm import tqdm
device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSNN_Layerwise:

    def __init__(self,device=device_local,is_pcn=True,is_svm=False):
        self.device = device
        self.timesteps = 15
        self.v_rest = 0
        self.v_thresh = 10
        self.v_reset = -1
        self.r_inhib = 3
        self.f_dep = 0.01
        self.lr = 0.01
        self.is_svm = is_svm
        self.is_pcn = is_pcn

        self.conv1 = CsnnLayer(self.input_shape(1),70,kernel_size=7,stride=1,padding=3,lr=self.lr,f_dep=self.f_dep,v_rest=self.v_rest,v_thresh=self.v_thresh,v_reset=self.v_reset,r_inhib=self.r_inhib,device=self.device)
        self.pool1 = SnnPooling(self.input_shape(70),kernel_size=3,stride=3,padding=0,timesteps=self.timesteps,device=self.device)
        if hasattr(self.conv1, 'weight'):
            self.conv1.weight = self.conv1.weight.to(self.device)

        if self.is_svm:
            self.svm_classifier = LinearSVC(dual="auto", max_iter=5000,C=0.005)
            self.is_svm_trained = False
        if self.is_pcn:
            self.pcn = ReadoutPCN(self.pool1.get_output_size(),10)
            if hasattr(self.pcn, 'weight'):
                self.pcn.weight = self.pcn.weight.to(self.device)



    def input_shape(self,channels):
        ##midify the size of each image here!##
        return 1,channels,28,28

    def forward(self,image_tensor,is_training=False,lr=0.01):
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        potential,spike=self.conv1.ttfs_inputlayer(image_tensor)
        potential,spike=self.conv1(potential,spike,is_training=is_training,lr=lr)
        potential,spike=self.pool1(potential,spike)

        return potential,spike

    def feature_extractor(self,image_tensor):
        potential,spike=self.forward(image_tensor)
        feature=spike.sum(dim=0).flatten().to(self.device)
        return feature

    def fit_svm(self,train_data,train_label):
        print(f">>> training SVM (num_samples: {len(train_label)})...")
        self.svm_classifier.fit(train_data,train_label)
        self.is_svm_trained=True

    def predict(self,image_tensor):
        if self.is_svm:
            feature=self.feature_extractor(image_tensor)
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy()
            return self.svm_classifier.predict(feature.reshape(1,-1))[0]
        if self.is_pcn:
            feature=self.feature_extractor_pcn(image_tensor)
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

    def fit_pcn(self, X_list, y_list, num_epochs=30):
        print(f"\n>>> PCN training，total Epochs: {num_epochs}")

        for epoch in range(num_epochs):

            X_shuffled, y_shuffled = shuffle(X_list, y_list)

            pbar = tqdm(range(len(X_shuffled)), desc=f"Epoch {epoch + 1}/{num_epochs}")

            for i in pbar:
                feat = X_shuffled[i].to(self.device)
                label = y_shuffled[i]

                self.pcn.s2_stdp_pcn_training(feat, label)

            if (epoch + 1) % 10 == 0:
                self.pcn.pos_lr *= 0.5
                self.pcn.neg_lr *= 0.5
                print(f"\n>>> lr decay: pos_lr: {self.pcn.pos_lr:.5f}, neg_lr: {self.pcn.neg_lr:.5f}")

        print(">>> PCN training complete!")

