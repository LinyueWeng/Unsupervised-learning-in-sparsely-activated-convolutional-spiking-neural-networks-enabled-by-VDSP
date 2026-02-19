import torch

class ReadoutPCN():
#only designed for batch size=1!!!
##input_shape is the size of the output from the last layer in 1D flattened vector (channel*height*width)##
    def __init__(self,
                 input_shape,
                 num_classes,
                 device='cpu',
                 t_max=1,
                 t_gap=0.1,
                 w_min=0.0,w_max=1.0,
                 pos_lr=0.01,
                 neg_lr=0.002,
                 weight_mean=0.5,weight_std=0.1
                 ):

        self.device = device
        self.num_classes = num_classes
        self.num_neurons = num_classes*2
        self.t_max=t_max
        self.w_min=w_min
        self.w_max=w_max
        self.t_gap=t_gap
        self.pos_lr=pos_lr
        self.neg_lr=neg_lr

        self.weight=torch.normal(mean=weight_mean,std=weight_std,size=(input_shape,self.num_neurons)).to(self.device)

    def forward(self,input_feature):
        if not isinstance(input_feature, torch.Tensor):
            input_feature = torch.tensor(input_feature, dtype=torch.float32)
        input_feature = input_feature.to(self.device)
        input_intensity=self.t_max-input_feature
        output_potential=torch.matmul(input_intensity,self.weight)
        output_latency=self.t_max*(1.0-output_potential/(output_potential.max()+ 1e-8))
        return output_latency#shape:(num_neurons)

    def s2_stdp_pcn_training(self,input_feature,label):
        ##weight update function##
        ##input_feature is the normalized latency of the first spike over all timesteps##

        if not isinstance(input_feature, torch.Tensor):
            input_feature = torch.tensor(input_feature, dtype=torch.float32)
        input_feature = input_feature.to(self.device)
        output_latency=self.forward(input_feature)
        t_mean=output_latency.mean()
        batch_size=1
        global_error = torch.zeros_like(output_latency).to(self.device)  # (num_neurons)

        #comput winner latency
        output_latency=output_latency.reshape(self.num_classes,2)#(num_classes,2)
        winner_indices=torch.argmin(output_latency,dim=1)#(num_classes)
        winner_latency=torch.gather(output_latency,dim=1,index=winner_indices.unsqueeze(1)).squeeze(1)#(num_classes)

        #comput error
        class_mask=torch.zeros_like(winner_latency).bool().to(self.device)
        class_mask[label]=True
        latency_target=torch.where(class_mask,t_mean-self.t_gap,t_mean+self.t_gap)
        error=(winner_latency-latency_target)/self.t_max#(num_classes)

        #generate global error matrix
        global_winner_indices=torch.arange(self.num_classes,device=self.device)*2+winner_indices#(num_classes)
        global_error.scatter_(0,global_winner_indices,error)

        #compute weight update
        input_trace=self.t_max-input_feature
        if input_trace.dim()==1:
            input_trace=input_trace.unsqueeze(0)
        if global_error.dim()==1:
            global_error=global_error.unsqueeze(0)
        grad=torch.matmul(input_trace.T,global_error)
        ltp_ltd=torch.where(grad>0,(self.w_max-self.weight)*self.pos_lr,(self.weight-self.w_min)*self.neg_lr)
        delta_weight=ltp_ltd*grad
        self.weight+=delta_weight
        self.weight=torch.clamp(self.weight,min=self.w_min,max=self.w_max)
        #print(f"delta_weight: {delta_weight.sum().item():.6f}")

    def predict(self,input_feature):
        output_latency=self.forward(input_feature)
        return output_latency.argmin()//2








