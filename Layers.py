import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import sys
from torch.nn.functional import conv2d, max_pool2d, unfold
from Synapse_Models import Ferroelectric
import matplotlib.pyplot as plt
import imageio
import io
from Characterization import ModelCharac
#device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_local = 'cpu'

Timesteps=15

class CsnnLayer:
    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_size=7,
                 stride=1,
                 padding=3,
                 lr=0.01,
                 f_dep=2,
                 timesteps=Timesteps,
                 w_min=0, w_max=1,
                 v_rest=0, v_thresh=10,v_reset=-1,
                 r_inhib=3, n_winners=7,
                 weight_mean=0.8, weight_std=0.05,
                 sfp=1.138,sfd=1.30,
                 v=1.0,
                 device=device_local,
                 synapse_model='Softbound'):

        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size, self.input_c, self.input_h, self.input_w = input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.timesteps = timesteps
        self.r_inhib = r_inhib
        self.lr = lr
        self.n_winners = n_winners
        self.sfp = sfp
        self.v = v
        self.sfd = sfd


        self.v_rest = torch.tensor(v_rest, dtype=torch.float32, device=self.device)
        self.v_thresh = torch.tensor(v_thresh, dtype=torch.float32, device=self.device)
        self.v_reset = torch.tensor(v_reset, dtype=torch.float32, device=self.device)
        self.w_min = torch.tensor(w_min, dtype=torch.float32, device=self.device)
        self.w_max = torch.tensor(w_max, dtype=torch.float32, device=self.device)
        self.f_dep = torch.tensor(f_dep, dtype=torch.float32, device=self.device)

        self.vdsp_cnt = 0
        self.update_lr_cnt = 500
        self.max_lr = 0.1
        self.adaptive_lr = True

        self.synapse_model_dictionary = {'Ferroelectric': Ferroelectric}
        self.synapse_model = synapse_model
        print(self.synapse_model)
        charac_model=ModelCharac(self.synapse_model)
        self.model_charac = charac_model()

        self.weight = torch.normal(mean=weight_mean, std=weight_std,
                                   size=(self.output_channels, self.input_c, self.kernel_size, self.kernel_size)).to(self.device)
        self.weight = torch.clamp(self.weight, min=self.w_min, max=self.w_max)

        self.output_h = (self.input_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_w = (self.input_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        self.potential = None
        self.activation = None
        self.vdsp_neurons = None

        self.target_channel = 0
        self.delta_weight_list = []
        self.frames = []
        self.update_counter=0

    def __call__(self, input_potential, input_spike, is_training=False):
        if not isinstance(input_potential, torch.Tensor):
            input_potential = torch.tensor(input_potential, dtype=torch.float32, device=self.device)
        else:
            input_potential = input_potential.to(self.device)

        if not isinstance(input_spike, torch.Tensor):
            input_spike = torch.tensor(input_spike, dtype=torch.float32, device=self.device)
        else:
            input_spike = input_spike.to(self.device)

        self.reset_state()


        if input_spike.dim() == 3:
            input_spike = input_spike.unsqueeze(0)
        if input_potential.dim() == 3:
            input_potential = input_potential.unsqueeze(0)

        timestep, c, h, w = input_spike.shape

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

            spike_mask = self.potential > self.v_thresh
            output_spike = torch.zeros_like(self.potential, dtype=torch.bool, device=self.device)

            if spike_mask.any():
                output_spike = self.lateral_inhibition_forward(spike_mask)

                if is_training:
                    winner_mask = self.get_vdsp_winners_mask()
                    delta_weight = 0
                    if winner_mask.any():
                        delta_weight=self.vsdp_batched(input_potential[t], input_spike[t], winner_mask)
                    """
                    Interpretability research. Unleash the next 2 functions to see the corresponding weight evolution.
                    one function a time only
                    """
                    #self.see_delta_weight_evolve(delta_weight)
                    #self.see_weight_frame()

                self.potential[output_spike] = self.v_reset
                self.activation[output_spike] = False

            output_spike_list.append(output_spike.clone())
            output_potential_list.append(self.potential.clone())

        return torch.stack(output_potential_list), torch.stack(output_spike_list)

    def lateral_inhibition_forward(self, spike_mask):
        max_pots, max_indices = self.potential.max(dim=0, keepdim=True)
        any_spike = spike_mask.any(dim=0, keepdim=True)

        output_spike = torch.zeros_like(spike_mask, device=self.device)
        output_spike.scatter_(0, max_indices, any_spike)

        inhibited_mask = any_spike & ~output_spike
        self.potential[inhibited_mask] = self.v_rest
        self.activation[inhibited_mask] = False

        return output_spike

    def get_vdsp_winners_mask(self):
        winner_mask = torch.zeros_like(self.potential, dtype=torch.bool, device=self.device)
        pots_tmp = self.potential * self.vdsp_neurons

        for _ in range(self.n_winners):
            max_val = pots_tmp.max()
            if max_val <= self.v_thresh:
                break

            flat_idx = torch.argmax(pots_tmp).item()
            c = flat_idx // (self.output_h * self.output_w)
            rem = flat_idx % (self.output_h * self.output_w)
            h = rem // self.output_w
            w = rem % self.output_w

            winner_mask[c, h, w] = True

            h_start = max(0, h - self.r_inhib)
            h_end = min(self.output_h, h + self.r_inhib + 1)
            w_start = max(0, w - self.r_inhib)
            w_end = min(self.output_w, w + self.r_inhib + 1)

            pots_tmp[:, h_start:h_end, w_start:w_end] = 0.0
            pots_tmp[c, :, :] = 0.0

        return winner_mask

    def vsdp_batched(self, input_potential_t, input_spike_t, winner_mask):
        win_c, win_h, win_w = torch.nonzero(winner_mask, as_tuple=True)
        N = len(win_c)
        if N == 0: return

        self.vdsp_cnt += N
        if self.adaptive_lr:
            old_period = (self.vdsp_cnt - N) // self.update_lr_cnt
            new_period = self.vdsp_cnt // self.update_lr_cnt
            if new_period > old_period:
                self.lr = min(self.lr * (2 ** (new_period - old_period)), self.max_lr)

        lr_t = torch.tensor(self.lr, dtype=torch.float32, device=self.device)

        pad = self.padding
        spk_unfold = unfold(input_spike_t.unsqueeze(0).float(), kernel_size=self.kernel_size, stride=self.stride,
                              padding=pad)
        pot_unfold = unfold(input_potential_t.unsqueeze(0).float(), kernel_size=self.kernel_size, stride=self.stride,
                              padding=pad)

        spatial_indices = (win_h * self.output_w + win_w).long()
        win_spikes = spk_unfold[0, :, spatial_indices].T.view(N, self.input_c, self.kernel_size, self.kernel_size)
        win_pots = pot_unfold[0, :, spatial_indices].T.view(N, self.input_c, self.kernel_size, self.kernel_size)

        w_winners = self.weight[win_c]

        if self.synapse_model == 'Softbound':
            cond_pot = win_spikes > 0
            w_factor = w_winners * (self.w_max - w_winners)
            g_dep = self.f_dep - (win_pots / self.v_thresh)

            delta_weights = torch.where(
                cond_pot,
                w_factor * 1.0 * lr_t,
                -w_factor * g_dep * lr_t
            )
        elif synapse_model_class := self.synapse_model_dictionary.get(self.synapse_model):
            cond_pot = win_spikes > 0
            synapse_model_instance = synapse_model_class(w_winners,v_ref=self.v,sf_p=self.sfp,sf_d=self.sfd,**self.model_charac)
            win_pots = torch.clamp(win_pots, min=0.0)
            delta_weights = synapse_model_instance(win_pots, self.v_thresh, cond_pot)
        else:
            raise ValueError(f"Synapse model {self.synapse_model} not found")

        self.weight.index_add_(0, win_c, delta_weights)
        self.weight.clamp_(self.w_min, self.w_max)

        self.vdsp_neurons[win_c, :, :] = False

        spatial_mask = torch.zeros((1, 1, self.output_h, self.output_w), dtype=torch.float32, device=self.device)
        spatial_mask[0, 0, win_h, win_w] = 1.0

        if self.r_inhib > 0:
            kernel_size = 2 * self.r_inhib + 1
            dilated_mask = max_pool2d(spatial_mask, kernel_size=kernel_size, stride=1, padding=self.r_inhib)
            dilated_mask = dilated_mask.squeeze(0).squeeze(0).bool()
        else:
            dilated_mask = spatial_mask.squeeze(0).squeeze(0).bool()

        self.vdsp_neurons[:, dilated_mask] = False

        return delta_weights

    def input_spike_generator(self, input_potential):
        input_spike = (input_potential >= self.v_thresh).float()
        if input_spike.dim() == 3:
            input_spike = input_spike.unsqueeze(0)
        return input_spike

    def ttfs_inputlayer(self, image_tensors):
        if not isinstance(image_tensors, torch.Tensor):
            image_tensors = torch.tensor(image_tensors, dtype=torch.float32, device=self.device)
        else:
            image_tensors = image_tensors.to(self.device)

        if image_tensors.dim() == 4:
            batch_size, image_c, image_h, image_w = image_tensors.shape
        elif image_tensors.dim() == 3:
            image_c, image_h, image_w = image_tensors.shape
        else:
            raise ValueError(f"Input dimension error: {image_tensors.shape}")

        images_spiketime = torch.floor((1.0 - image_tensors) * self.timesteps)

        steps = torch.arange(1, self.timesteps + 1, dtype=torch.float32, device=self.device).view(-1, 1, 1, 1)
        input_potential = steps * self.v_thresh / (images_spiketime + 1.0)
        input_potential = torch.clamp(input_potential, max=self.v_thresh.item())
        input_spike = self.input_spike_generator(input_potential)
        return input_potential, input_spike

    def reset_state(self):
        self.potential = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.float32,
                                    device=self.device) * self.v_rest
        self.activation = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.bool,
                                     device=self.device)
        self.vdsp_neurons = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.bool,
                                       device=self.device)

    def get_output_size(self):
        return self.output_channels * self.output_h * self.output_w

    def see_delta_weight_evolve(self, delta_weight):
        if isinstance(delta_weight, torch.Tensor):
            current_val = delta_weight.mean().item()
        else:
            current_val = float(delta_weight)
        self.delta_weight_list.append(current_val)

        plt.figure(figsize=(10, 5))
        plt.plot(self.delta_weight_list, color='blue', marker='o', markersize=3)
        plt.title(f"Delta Weight Evolution (Step {len(self.delta_weight_list)})")
        plt.xlabel("Timesteps")
        plt.ylabel("Sum of Delta Weights")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        self.frames.append(frame)
        plt.close()

        if len(self.delta_weight_list) == 10*self.timesteps:
            gif_path = 'delta_evolution.gif'
            imageio.mimsave(gif_path, self.frames, fps=5)
            print(f"GIF saved to: {gif_path}")
            input("Weight tracking image saved. Press Enter to exit...")
            sys.exit()

    def see_weight_frame(self,):

        self.update_counter += 1

        if self.update_counter % 10 != 0:
            return
        w_map = self.weight[0, 0].cpu().numpy()

        fig, ax = plt.subplots()
        im = ax.imshow(w_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(im)
        ax.set_title(f"Weight Frame {len(self.frames) + 1}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(imageio.v2.imread(buf))
        plt.close()

        if len(self.frames) >= 1000:
            gif_path = 'weight_evolution_grid.gif'
            imageio.mimsave(gif_path, self.frames, fps=50)
            print(f"\nGIF saved to: {gif_path}")

            sys.exit()

class SnnPooling:
    ## max pooling layer for spiking neural network ##
    ## standard pooling layer with outputs: output_potential and output_spike, all in the shape of [timesteps, output_channels, output_h, output_w]##
    def __init__(self,input_shape,
                 kernel_size=3,
                 stride=1,
                 padding=3,
                 v_thresh=10,
                 v_reset=-1,
                 timesteps=15,
                 device=device_local
                 ):
        self.device=device
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.v_thresh=v_thresh
        self.v_reset=v_reset
        self.timesteps=timesteps

        batch_size,self.input_c,input_h,input_w=input_shape

        self.output_h = (input_h - kernel_size + 2 * padding) // stride + 1
        self.output_w = (input_w - kernel_size + 2 * padding) // stride + 1

        self.activation=None

    def __call__(self,input_potential,input_spike):
        self.reset_state()
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
