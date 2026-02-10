import torch

from torch.nn.functional import conv2d,unfold,max_pool2d

device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CsnnLayer:

    def __init__(self,input_shape,
                 output_channels,
                 kernel_size=7,
                 stride=1,
                 padding=3,
                 lr=0.01,
                 f_dep=2,
                 timesteps=15,
                 w_min=0,w_max=1,
                 v_rest=0,
                 v_thresh=10,
                 v_reset=-1,
                 r_inhib=3,
                 weight_mean=0.8,weight_std=0.05,
                 device=device_local
                 ):

        self.device=device
        self.batch_size,self.input_c,self.input_h,self.input_w=input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.timesteps = timesteps
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.w_min = w_min
        self.w_max = w_max
        self.r_inhib = r_inhib
        self.lr=lr
        self.f_dep=f_dep

        self.weight=torch.normal(mean=weight_mean, std=weight_std,size=(self.output_channels,self.input_c,self.kernel_size,self.kernel_size)).to(device)
        self.weight=torch.clamp(self.weight, min=self.w_min, max=self.w_max).to(self.device)

        self.output_h = (self.input_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_w = (self.input_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        self.potential = torch.ones(self.output_channels,self.output_h,self.output_w,device=self.device)*self.v_rest

        self.activation = torch.ones(self.output_channels,self.output_h,self.output_w).bool().to(self.device)


    def __call__(self,input_potential,input_spike,is_training=False,lr=0.01):
        self.reset_state()
        self.lr=lr
        if input_spike.dim()==3:
            input_spike=input_spike.unsqueeze(0)
        if input_potential.dim()==3:
            input_potential=input_potential.unsqueeze(0)
        output_spike_list=[]
        output_potential_list=[]
        timestep,c,h,w=input_spike.shape

        for t in range(timestep):
            potential_update=conv2d(input_spike[t].unsqueeze(0).float(),self.weight,stride=self.stride,padding=self.padding)[0]
            self.potential[self.activation]+=potential_update[self.activation]

            output_spike = torch.zeros_like(self.potential).bool().to(self.device)

            spike_mask=self.potential>=self.v_thresh
            if spike_mask.any():
                winner=self.lateral_inhibition(spike_mask)
                self.potential[spike_mask]=self.v_reset
                self.activation[spike_mask]=False
                output_spike[winner]=True
                if is_training:
                    self.vsdp(input_potential[t],input_spike[t],winner)
            output_spike_list.append(output_spike.clone())
            output_potential_list.append(self.potential.clone())

        return torch.stack(output_potential_list).to(self.device),torch.stack(output_spike_list).to(self.device)

    def lateral_inhibition(self,spike_mask):
        winner = torch.zeros_like(self.potential).bool().to(self.device)
        spike_c,spike_h,spike_w=torch.nonzero(spike_mask,as_tuple=True)
        spike_potential=self.potential[spike_c,spike_h,spike_w]
        argsort_spike_mask = torch.argsort(spike_potential,descending=True).to(self.device)
        for i in argsort_spike_mask:
            if self.potential[spike_c[i],spike_h[i],spike_w[i]]>self.v_reset:
                self.potential[:,spike_h[i],spike_w[i]]=self.v_reset
                self.activation[:,spike_h[i],spike_w[i]]=False
                winner[spike_c[i],spike_h[i],spike_w[i]]=True
                #local inter-map competition
                h_floor=max(0,spike_h[i]-self.r_inhib)
                w_floor=max(0,spike_w[i]-self.r_inhib)
                h_ceil=min(spike_h[i]+self.r_inhib+1,self.output_h)
                w_ceil=min(spike_w[i]+self.r_inhib+1,self.output_w)
                self.activation[:,h_floor:h_ceil,w_floor:w_ceil]=False
                self.potential[:, h_floor:h_ceil,w_floor:w_ceil] = self.v_reset

                # global intra-map competition
                self.activation[spike_c[i], :, :] = False
                self.potential[spike_c[i], :, :] = self.v_reset

        return winner

    def vsdp(self, input_potential, input_spike, winner):
        winner_c, winner_h, winner_w = torch.nonzero(winner, as_tuple=True)
        if len(winner_c) == 0:
            return 0

        input_spike_unfolded = unfold(input_spike.unsqueeze(0).float(),
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        input_pot_unfolded = unfold(input_potential.unsqueeze(0).float(),
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        for index in range(len(winner_c)):
            c_pos = winner_c[index].item()
            h_pos = winner_h[index].item()
            w_pos = winner_w[index].item()
            receptive_index = h_pos * self.output_w + w_pos

            win_spike = input_spike_unfolded[0, :, receptive_index].view(self.input_c, self.kernel_size,
                                                                         self.kernel_size)
            win_pot = input_pot_unfolded[0, :, receptive_index].view(self.input_c, self.kernel_size, self.kernel_size)

            cond_pot = win_spike > 0

            cond_dep = ~cond_pot

            w_factor = self.weight[c_pos] * (self.w_max - self.weight[c_pos])

            delta_weight = torch.zeros_like(self.weight[c_pos])

            #LTP
            delta_weight[cond_pot] += w_factor[cond_pot] * self.lr

            #LTD
            if cond_dep.any():
                norm_potential = win_pot[cond_dep] / self.v_thresh

                g_dep = self.f_dep - norm_potential

                delta_weight[cond_dep] += w_factor[cond_dep] * g_dep * self.lr

            self.weight[c_pos] += delta_weight
            self.weight[c_pos].clamp_(self.w_min, self.w_max)

            if index % 50 == 0:
                print(
                    f"Delta: {delta_weight.sum().item():.6f} (LTP: {delta_weight[cond_pot].sum():.6f}, LTD: {delta_weight[cond_dep].sum():.6f})")

        return self.weight
    def input_spike_generator(self,input_potential):
        input_spike=input_potential>=self.v_thresh
        if not isinstance(input_spike,torch.Tensor):
            input_spike=torch.tensor(input_spike.float()).to(self.device)
        if input_spike.dim() == 3:
            input_spike = input_spike.unsqueeze(0)
        return input_spike

    def ttfs_inputlayer(self,image_tensors):

        if image_tensors.dim() == 4:
            batch_size, image_c, image_h, image_w = image_tensors.shape

        elif image_tensors.dim() == 3:
            image_c, image_h, image_w = image_tensors.shape

        else:
            raise ValueError(f"input dimension error: {image_tensors.shape}")

        images_spiketime = torch.floor((1 - image_tensors) * self.timesteps).to(self.device)
        input_potential = torch.zeros(self.timesteps, image_c, image_h, image_w).to(self.device)
        for i in range(self.timesteps):
            input_potential[i] = (i+1)*self.v_thresh/(images_spiketime+1)

        input_spike=self.input_spike_generator(input_potential)
        return input_potential,input_spike

    def reset_state(self):
        self.potential = torch.ones(self.output_channels, self.output_h, self.output_w,device=self.device) * self.v_rest
        self.activation = torch.ones(self.output_channels, self.output_h, self.output_w).bool().to(self.device)


class SnnPooling:
    def __init__(self,input_shape,
                 kernel_size=3,
                 stride=1,
                 padding=3,
                 f_dep=0.01,
                 timesteps=15,
                 w_min=0,w_max=1,
                 v_rest=0,
                 v_thresh=10,
                 v_reset=-1,
                 r_inhib=3,
                 device=device_local
                 ):
        self.device=device
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.f_dep=f_dep
        self.v_rest=v_rest
        self.v_thresh=v_thresh
        self.v_reset=v_reset
        self.r_inhib=r_inhib
        self.w_min=w_min
        self.w_max=w_max
        self.timesteps=timesteps

        batch_size,self.input_c,input_h,input_w=input_shape

        self.output_h = (input_h - kernel_size + 2 * padding) // stride + 1
        self.output_w = (input_w - kernel_size + 2 * padding) // stride + 1

        self.activation=None

    def __call__(self,input_potential,input_spike):
        self.reset_state()
        timestep,c,h,w=input_spike.shape
        output_spike_list=[]
        output_potential_list=[]

        self.activation = torch.ones(self.input_c,self.output_h,self.output_w).bool().to(self.device)

        for t in range(timestep):
            output_spike=input_spike[t].clone()
            output_potential=input_potential[t].clone()
            compensate_mask=input_spike[t] > 0
            output_potential[compensate_mask]=self.v_thresh

            output_spike=max_pool2d(output_spike.unsqueeze(0).float(),kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)[0].bool()
            output_potential=max_pool2d(output_potential.unsqueeze(0).float(),kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)[0]


            spike_mask= output_spike & self.activation
            self.activation[spike_mask]=False
            output_potential[spike_mask]=self.v_reset


            output_spike_list.append(spike_mask.float())
            output_potential_list.append(output_potential.clone())
        return torch.stack(output_potential_list).to(self.device),torch.stack(output_spike_list).to(self.device)

    def reset_state(self):
        self.activation=None
