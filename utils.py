import torch
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

save_path = "C:/Users/28218/PycharmProjects/CSNN/figures"

class DoGTransform:
    """
    Not used. Please ignore it if you are reproducing.
    """
    def __init__(self, device='cpu',kernel_size=7, sigma1=1.0, sigma2=2.0):
        self.kernel_size = kernel_size
        self.device = device
        self.g1 = self.get_gaussian_kernel(kernel_size, sigma1).to(self.device)
        self.g2 = self.get_gaussian_kernel(kernel_size, sigma2).to(self.device)

    def get_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size).float() - kernel_size // 2
        x = torch.exp(-x ** 2 / (2 * sigma ** 2))
        x = x / x.sum()
        kernel = x.view(1, 1, -1, 1) * x.view(1, 1, 1, -1)
        return kernel

    def __call__(self, img):
        if img.dim() == 2:
            img_batch = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            img_batch = img.unsqueeze(0)
        elif img.dim() == 4:
            img_batch = img
        else:
            raise ValueError(f"unsupported image dimensions: {img.dim()}")

        blur1 = conv2d(img_batch, self.g1, padding=self.kernel_size // 2)
        blur2 = conv2d(img_batch, self.g2, padding=self.kernel_size // 2)

        dog = blur1 - blur2

        on_center = torch.clamp(dog, min=0)
        off_center = torch.clamp(-dog, min=0)

        on_center = on_center / (on_center.max() + 1e-5)
        off_center = off_center / (off_center.max() + 1e-5)

        return torch.cat([on_center, off_center], dim=1).squeeze(0)

def plot_weight_histogram(pth_file_path, bins=1000, smooth_sigma=5.0, prominence_threshold=0.08):
    """
    plot the weight histogram of the given checkpoint file, and detect the attractors.
    used only during the project for temporary presentation, not used in the thesis.
    parameters:
    - smooth_sigma: gaussian smooth, normally no need to change
    - prominence_threshold: increase this value to detect attractors with stricter, decrease if you can see attractors but they cannot be detected.
    """
    if not os.path.exists(pth_file_path):
        print(f"no such file: {pth_file_path}")
        return

    print(f"loading: {pth_file_path} ...")
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    all_weights = []

    #handle all different types of checkpoints
    if isinstance(checkpoint, torch.Tensor):
        all_weights.append(checkpoint.detach().numpy().flatten())
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor) and param.dim() > 1:
                all_weights.append(param.detach().numpy().flatten())
    elif isinstance(checkpoint, list):
        for param in checkpoint:
            if isinstance(param, torch.Tensor):
                all_weights.append(param.detach().numpy().flatten())

    if not all_weights:
        print("no data detected in the checkpoint, please check the file format or the model structure.")
        return

    global_weights = np.concatenate(all_weights)

    # filter out the weights that are too small or too large, which may be caused by torch.clamp()
    valid_weights = global_weights[(global_weights > 0.001) & (global_weights < 0.999)]

    counts, bin_edges = np.histogram(valid_weights, bins=bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_counts = gaussian_filter1d(counts, sigma=smooth_sigma)

    max_count = np.max(smoothed_counts)
    normalized_counts = smoothed_counts / max_count

    #add 0 to the beginning and end of the array to make sure the peak can be detected
    padded_counts = np.pad(normalized_counts, (1, 1), 'constant', constant_values=(0, 0))

    padded_peaks, properties = find_peaks(padded_counts, prominence=prominence_threshold)

    #eliminate the index introduced by padding
    peaks = padded_peaks - 1

    peaks = peaks[(peaks >= 0) & (peaks < len(normalized_counts))]

    print("-" * 50)
    print(f" {len(peaks)} Attractors are detected:")

    #attractors
    sorted_indices = np.argsort(normalized_counts[peaks])[::-1]
    attractors = []

    for i, peak_idx in enumerate(peaks[sorted_indices]):
        weight_val = bin_centers[peak_idx]
        relative_height = normalized_counts[peak_idx]
        attractors.append(weight_val)
        print(f"  [{i + 1}] mean: {weight_val:.4f} (relative prominence: {relative_height:.2%})")
    print("-" * 50)

    #plot
    plt.figure(figsize=(10, 6))

    plt.bar(bin_centers, counts / np.max(counts), width=1 / bins, color='lightgray', alpha=1, label='Raw Histogram')

    plt.plot(bin_centers, normalized_counts, color='royalblue', linewidth=2, label='Smoothed Distribution')

    for peak_idx in peaks:
        plt.plot(bin_centers[peak_idx], normalized_counts[peak_idx], "rx", markersize=10, markeredgewidth=2)
        plt.axvline(x=bin_centers[peak_idx], color='red', linestyle='--', alpha=0.5)
        plt.text(bin_centers[peak_idx] + 0.02, normalized_counts[peak_idx],
                 f"{bin_centers[peak_idx]:.3f}", color='red', fontsize=11, fontweight='bold')

    plt.title("Multi-Attractor Detection in Weight Distribution", fontsize=14)
    plt.xlabel("Weight Value", fontsize=12)
    plt.ylabel("Normalized Frequency", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return attractors

def plot_parameter_fitting_grid(experiment_data,overdrive_ratios,beta_values,xlabel,ylabel,title):
    """
    heat map of the parameter fitting grid in the Exponential Model. Fig.4.1
    :param experiment_data:
    :param overdrive_ratios:
    :param beta_values:
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    """
    y_labels = [f"{ov}\n{b}" for ov, b in zip(overdrive_ratios, beta_values)]

    v_refs = sorted(list(experiment_data.keys()))

    num_rows = len(overdrive_ratios)
    num_cols = len(v_refs)

    accuracy_matrix = np.zeros((num_rows, num_cols))
    sfp_annotation_matrix = np.empty((num_rows, num_cols), dtype=object)

    for col_idx, v_ref in enumerate(v_refs):
        data = experiment_data[v_ref]
        for row_idx in range(num_rows):
            accuracy_matrix[row_idx, col_idx] = data['mean'][row_idx]
            sfp_annotation_matrix[row_idx, col_idx] = f"{data['sfp'][row_idx]:.4f}"

    plt.figure(figsize=(10, 8), dpi=120)

    ax = sns.heatmap(
        accuracy_matrix,
        annot=sfp_annotation_matrix,
        fmt='',
        cmap='RdYlGn',
        cbar_kws={'label': 'Accuracy Mean (%)'},
        linewidths=1,
        linecolor='white'
    )

    ax.set_xticks(np.arange(num_cols) + 0.5)
    ax.set_xticklabels([f"{v} V" for v in v_refs], fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    ax.set_yticks(np.arange(num_rows) + 0.5)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    plt.title(title,
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    plt.savefig(f"{save_path}/exponential_model_heat_map.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{save_path}/exponential_model_heat_map.png", dpi=300, bbox_inches="tight", )
    plt.show()


data5 = {
    1.0: {
        'sfp': [
            1.0113, 1.0283, 1.0377, 1.0452, 1.0465, 1.0565, 1.0620,
            1.0824, 1.0848, 1.0889, 1.1130, 1.1131, 1.1695, 1.1833
        ],
        'mean': [
            98.6160, 98.7720, 98.6680, 98.6800, 98.6240, 98.6520, 98.5120,
            98.4400, 98.4760, 98.4880, 98.5360, 98.5560, 98.5600, 98.4480
        ],
    },
    1.2: {
        'sfp': [1.0500, 1.0862, 1.1000, 1.1150, 1.1300, 1.1500, 1.1800, 1.2100, 1.2400, 1.2700, 1.3000, 1.3500, 1.4200, 1.5000],
        'mean': [98.528, 98.584, 98.621, 98.452, 98.476, 98.492, 98.544, 98.444, 98.412, 98.54, 98.548, 98.352, 98.448, 98.404]
    },
    1.5: {
         'sfp': [1.15, 1.1863, 1.2187, 1.24, 1.28, 1.35, 1.4385, 1.55, 1.6578, 1.8, 1.877, 2.05, 2.15, 2.3155],
         'mean': [98.40399932861328, 98.42799377441406, 98.27200317382812, 98.49999237060547, 98.39601135253906, 98.47200775146484, 98.51599884033203, 98.37600708007812, 98.20800018310547, 98.36399841308594, 98.29600524902344, 98.03999328613281, 98.31600189208984, 98.24400329589844]
     },

}# data for Fig.4.1, heat map of compensated solution



overdrive_values5 = [' ', ' ',' ' ,' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
beta_values5 = ['', '(β≈1.2)', '(β≈(1.05)', '', '', '', '', '', '', '', '', '', '', '']
xlabel4='Reference Voltage (V)'
ylabel4='Overdrive Ratio & $\\beta$ Value'
title4='Accuracy Mapping: $V_{ref}$ vs. Overdrive ($\\beta$) with $sf_p$ Values'





weights_path="C:/Users/28218/PycharmProjects/CSNN/checkpoints_CSNN/snn_weight_EM.pth"
if __name__ == "__main__":
    #plot_weight_histogram(weights_path)
    plot_parameter_fitting_grid(data5, overdrive_values5, beta_values5, xlabel4, ylabel4, title4)


