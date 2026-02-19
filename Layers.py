import torch

from torch.nn.functional import conv2d,unfold,max_pool2d

device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CsnnLayer:
    ## convolutional layer for spiking neural network ##
    ## standard convolutional layer with outputs: output_potential and output_spike, all in the shape of [timesteps, output_channels, output_h, output_w]##
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
        timestep,c,h,w=input_spike.shape

        all_potential_updates = conv2d(
            input_spike.float(),
            self.weight,
            stride=self.stride,
            padding=self.padding
        )

        output_spike_list = []
        output_potential_list = []

        for t in range(timestep):
            potential_update = all_potential_updates[t]

            self.potential[self.activation] += potential_update[self.activation]

            output_spike = torch.zeros_like(self.potential, dtype=torch.bool, device=self.device)
            spike_mask = self.potential >= self.v_thresh

            if spike_mask.any():
                winner = self.lateral_inhibition(spike_mask)

                self.potential[spike_mask] = self.v_reset
                self.activation[spike_mask] = False

                output_spike[winner] = True

                if is_training:
                    self.vsdp(input_potential[t], input_spike[t], winner)

            output_spike_list.append(output_spike.clone())
            output_potential_list.append(self.potential.clone())

        return torch.stack(output_potential_list), torch.stack(output_spike_list)

    def lateral_inhibition(self,spike_mask):
        ##use Fast NMS and pooling operator to speed up the lateral inhibition process (mainly for GPU)##

        winner = torch.zeros_like(self.potential).bool().to(self.device)
        spike_c,spike_h,spike_w=torch.nonzero(spike_mask,as_tuple=True)
        spike_potential=self.potential[spike_c,spike_h,spike_w]
        argsort_spike_mask = torch.argsort(spike_potential,descending=True).to(self.device)
        C = spike_c[argsort_spike_mask]
        H = spike_h[argsort_spike_mask]
        W = spike_w[argsort_spike_mask]
        num_spikes = len(C)

        if num_spikes > 0:
            keep = torch.ones(num_spikes, dtype=torch.bool, device=self.device)

            for i in range(num_spikes):
                if not keep[i]:
                    continue

                conflict_c = (C[i + 1:] == C[i])

                conflict_hw = (torch.abs(H[i + 1:] - H[i]) <= self.r_inhib) & \
                              (torch.abs(W[i + 1:] - W[i]) <= self.r_inhib)

                keep[i + 1:][conflict_c | conflict_hw] = False

            win_c = C[keep]
            win_h = H[keep]
            win_w = W[keep]

            winner[win_c, win_h, win_w] = True

            self.activation[win_c] = False
            self.potential[win_c] = self.v_reset

            spatial_mask = torch.zeros((self.output_h, self.output_w), device=self.device)
            spatial_mask[win_h, win_w] = 1.0  # 点亮赢家所在的位置

            if self.r_inhib > 0:
                kernel_size = 2 * self.r_inhib + 1
                expanded_mask = max_pool2d(
                    spatial_mask.unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=self.r_inhib
                ).squeeze(0).squeeze(0).bool()
            else:
                expanded_mask = spatial_mask.bool()

            self.activation[:, expanded_mask] = False
            self.potential[:, expanded_mask] = self.v_reset

        return winner

    def vsdp(self, input_potential, input_spike, winner):

        winner_c, winner_h, winner_w = torch.nonzero(winner, as_tuple=True)
        if len(winner_c) == 0:
            return 0

        input_spike_unfolded = unfold(input_spike.unsqueeze(0).float(),
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        input_pot_unfolded = unfold(input_potential.unsqueeze(0).float(),
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        N = len(winner_c)
        if N == 0:
            return self.weight

        receptive_indices = winner_h * self.output_w + winner_w

        win_spikes = input_spike_unfolded[0, :, receptive_indices].T.view(
            N, self.input_c, self.kernel_size, self.kernel_size)
        win_pots = input_pot_unfolded[0, :, receptive_indices].T.view(
            N, self.input_c, self.kernel_size, self.kernel_size)

        w_winners = self.weight[winner_c]

        cond_pot = win_spikes > 0  # LTP
        w_factor = w_winners * (self.w_max - w_winners)

        norm_pots = win_pots / self.v_thresh
        g_dep = self.f_dep - norm_pots

        delta_weights = torch.where(
            cond_pot,
            w_factor * self.lr,  # LTP
            w_factor * g_dep * self.lr  # LTD
        )

        self.weight.index_add_(dim=0, index=winner_c, source=delta_weights)

        self.weight.clamp_(self.w_min, self.w_max)

        # if N > 0:
        #     print(f"Total Delta: {delta_weights.sum().item():.6f} "
        #           f"(LTP: {delta_weights[cond_pot].sum():.6f}, "
        #           f"LTD: {delta_weights[~cond_pot].sum():.6f})")

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

    def get_output_size(self):#if you connect it to readout layer, use this function to get the input size of the pcn layer
        return self.output_channels*self.output_h*self.output_w


class SnnPooling:
    ## max pooling layer for spiking neural network ##
    ## standard pooling layer with outputs: output_potential and output_spike, all in the shape of [timesteps, output_channels, output_h, output_w]##
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

        comp_potential = input_potential.clone()
        comp_potential[input_spike > 0] = self.v_thresh
        pooled_spike = max_pool2d(input_spike.float(), kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding).bool()
        pooled_potential = max_pool2d(comp_potential.float(), kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding)
        cumsum_spikes = torch.cumsum(pooled_spike.int(), dim=0)
        spike_mask = pooled_spike & (cumsum_spikes == 1) & self.activation.unsqueeze(0)
        final_potential = pooled_potential.clone()
        final_potential[spike_mask] = self.v_reset
        has_spiked = cumsum_spikes[-1] > 0
        self.activation = self.activation & (~has_spiked)
        return final_potential.to(self.device), spike_mask.float().to(self.device)

    def reset_state(self):
        self.activation=None

    def get_output_size(self):#if you connect it to readout layer, use this function to get the input size of the pcn layer
        return self.input_c*self.output_h*self.output_w
