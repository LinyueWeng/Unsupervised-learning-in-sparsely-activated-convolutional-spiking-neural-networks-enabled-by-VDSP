import torch

from torch.nn.functional import conv2d,unfold,max_pool2d

from Layers import CsnnLayer,SnnPooling
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSNN_Layerwise:

    def __init__(self,device=device_local):
        self.device = device
        self.timesteps = 15
        self.v_rest = 0
        self.v_thresh = 10
        self.v_reset = -1
        self.r_inhib = 3
        self.f_dep = 0.01
        self.lr = 0.01

        self.conv1 = CsnnLayer(self.input_shape(1),70,kernel_size=7,stride=1,padding=3,lr=self.lr,f_dep=self.f_dep,v_rest=self.v_rest,v_thresh=self.v_thresh,v_reset=self.v_reset,r_inhib=self.r_inhib,device=self.device)
        self.pool1 = SnnPooling(self.input_shape(70),kernel_size=3,stride=3,padding=0,timesteps=self.timesteps,device=self.device)
        if hasattr(self.conv1, 'weight'):
            self.conv1.weight = self.conv1.weight.to(device)

        self.svm_classifier = LinearSVC(dual="auto", max_iter=5000,C=0.005)
        self.is_svm_trained = False


    def input_shape(self,channels):
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
        print(f">>> 开始训练 SVM 层 (样本数: {len(train_label)})...")
        self.svm_classifier.fit(train_data,train_label)
        self.is_svm_trained=True

    def predict(self,image_tensor):
        feature=self.feature_extractor(image_tensor)
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        return self.svm_classifier.predict(feature.reshape(1,-1))[0]

