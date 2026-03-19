import torch
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

class DoGTransform:
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

def accuracy_samples_analysis(data_matrix,metadata):
    plt.figure(figsize=(12, 7))

    # Get unique group IDs from the first column of the matrix
    unique_groups = np.unique(data_matrix[:, 0])

    for group_id in unique_groups:
        group_mask = data_matrix[:, 0] == group_id
        subset = data_matrix[group_mask]

        x_samples = subset[:, 1]
        y_means = subset[:, 2]
        y_stds = subset[:, 3]

        meta = metadata[group_id]

        plt.errorbar(x_samples, y_means, yerr=y_stds,
                     fmt=meta['marker'] + '-',
                     color=meta['color'],
                     label=meta['label'],
                     capsize=4, linewidth=1.5)

        # for x, y in zip(x_samples, y_means):
        #     plt.annotate(f'{y:.2f}', (x, y),
        #                  textcoords="offset points",
        #                  xytext=(0, meta['offset']),
        #                  ha='center', fontsize=9,
        #                  color=meta['color'], fontweight='bold')

    plt.title('Automated Accuracy Plot from Data Matrix', fontsize=14)
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)

    plt.xscale('symlog', linthresh=1000)
    #tick_values = np.unique(data_matrix[:, 1])
    tick_values= torch.arange(0,1000,100)
    tick_values= torch.cat((tick_values,torch.arange(0,10000,1000)))
    tick_values = torch.cat((tick_values, torch.arange(10000, 60000, 10000)))
    plt.xticks(tick_values, [str(int(v)) for v in tick_values], rotation=45)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.ylim(97, 99)

    plt.tight_layout()
    plt.show()

def plot_weight_histogram(pth_file_path, bins=1000, smooth_sigma=5.0, prominence_threshold=0.08):
    """
    plot the weight histogram of the given checkpoint file, and detect the attractors.

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
    y_labels = [f"{ov}\n(β≈{b})" for ov, b in zip(overdrive_ratios, beta_values)]

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

    plt.savefig('accuracy_heatmap.png')
    plt.show()

def plot_twiny():
    v_ref = ['1.0V', '1.2V', '1.5V']
    samples = [3223, 1045, 421]
    accuracy = [98.77, 98.58, 98.43]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    plt.rcParams.update({'font.size': 12})

    color_samples = 'tab:blue'
    lns1 = ax1.plot(v_ref, samples, marker='o', color=color_samples, label='Number of training samples')
    ax1.set_xlabel('Reference Potential (V)')
    ax1.set_ylabel('Number of training samples')

    ax2 = ax1.twinx()

    color_acc = 'tab:orange'
    lns2 = ax2.plot(v_ref, accuracy, linestyle='--', color=color_acc, label='Accuracy')
    ax2.set_ylabel('Accuracy (%)')

    ax2.set_ylim(min(accuracy) - 0.5, max(accuracy) + 0.5)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    plt.tight_layout()
    plt.show()



weights_path="C:/Users/28218/PycharmProjects/CSNN/checkpoints_CSNN/snn_weight_epoch_1.pth"

data1 = np.array([
    # Linear (PCN) - Group 0
    [0, 0, 93.17, 0.69],
    [0, 126, 96.07, 0.37],
    [0, 157, 95.97, 0.42],
    [0, 210, 96.06, 0.30],
    [0, 566, 96.25, 0.29],
    [0, 644, 95.74, 0.18],
    [0, 828, 96.03, 0.32],
    [0, 1268, 96.00, 0.31],

    # Linear (SVM)% - Group 1
    [1, 0, 97.30, 0.20],
    [1, 126, 98.43, 0.13],
    [1, 157, 98.50, 0.21],
    [1, 210, 98.66, 0.19],
    [1, 566, 98.29, 0.13],
    [1, 644, 98.45, 0.12],
    [1, 828, 98.50, 0.24],
    [1, 1268, 98.30, 0.13]
])#CSNN accuracy with SVM and PCN
data2 = np.array([
    # --- Group 2: Linear (PCN)% ---
    [2, 0,   8.81,  0.67],
    [2, 1,   87.09, 0.41],
    [2, 2,   91.30, 0.45],
    [2, 3,   91.48, 0.27],
    [2, 5,   93.05, 0.42],
    [2, 8,   93.58, 0.57],
    [2, 10,  94.50, 0.47],
    [2, 15,  94.75, 0.34],
    [2, 30,  95.07, 0.40],
    [2, 40,  95.47, 0.31],
    [2, 50,  96.11, 0.45],
    [2, 60,  95.33, 0.37],
    [2, 70,  95.32, 0.44],
    [2, 80,  95.50, 0.24],
    [2, 90,  96.08, 0.36],
    [2, 100, 96.32, 0.33],
    [2, 150, 96.06, 0.29],
    [2, 200, 96.10, 0.36]
])#PCN accuracy with CSNN
data3 = np.array([
    [3, 0,     97.32, 0.36],
    [3, 231,   97.70, 0.26],
    [3, 990,   98.27, 0.20],
    [3, 1946,  98.41, 0.18],
    [3, 2985,  98.29, 0.17],
    [3, 4609,  98.40, 0.13],
    [3, 6766,  98.19, 0.22],
    [3, 14136, 98.11, 0.13],
    [3, 31712, 98.15, 0.23],
    [4, 0,     97.34, 0.18],
    [4,492,98.53,0.15],
    [4, 589,   98.61, 0.13],
    [4, 1269,  98.43, 0.20],
    [4, 2350,  98.63, 0.15],
    [4, 4639,  98.58, 0.23],
    [4, 7388,  98.53, 0.22],
    [4, 13102, 98.41, 0.24],
    [4, 37763, 98.36, 0.31],
    [4, 60000, 98.50, 0.27],
    [1, 0, 97.30, 0.20],
    [1, 126, 98.43, 0.13],
    [1, 157, 98.50, 0.21],
    [1, 210, 98.66, 0.19],
    [1, 566, 98.29, 0.13],
    [1, 644, 98.45, 0.12],
    [1, 828, 98.50, 0.24],
    [1, 1268, 98.30, 0.13],
    [1, 1897, 98.49, 0.26],
    [1, 3468, 98.47, 0.16],
    [1, 11482, 98.32, 0.24],
    [1, 15639, 98.29, 0.15],

])#parameter fitting accuracy curve
group_metadata = {
    0: {"label": "Softbound (PCN)", "color": "#1f77b4", "marker": "o", "offset": -15},
    1: {"label": "Softbound (SVM)", "color": "#d62728", "marker": "s", "offset": 10},
    2: {"label": "Softbound (PCN) - Epochs", "color": "#2ca02c","marker": "^", "offset": 12},
    3: {"label": "Ferroelectric (SVM, Not Optimal)", "color": "#e377c2", "marker": "d", "offset": 0},
    4: {"label": "Ferroelectric (SVM, Optimal)", "color": "#2ca02c", "marker": "v", "offset": 0}
}




data4 = {
    1.0: {
        'sfp': [1.0277, 1.06925, 1.108, 1.1385, 1.2077, 1.277, 1.4155],
        'mean':[98.5440, 98.7280, 98.5200, 98.4600, 98.3920, 98.5800, 98.4360]
    },
    1.2: {
        'sfp': [1.0517, 1.1292, 1.2068, 1.2585, 1.3877, 1.5170, 1.7755],
        'mean': [98.6880, 98.6440, 98.6000, 98.5160, 98.3720, 98.0080, 98.1440]
    },
    1.5: {
        'sfp': [1.0877, 1.2192, 1.3508, 1.4385, 1.6578, 1.8770, 2.3155],
        'mean': [98.5640, 98.5000, 98.4880, 98.6600, 98.1560, 98.1080, 98.2920]
    },
    1.8: {
        'sfp': [1.1237, 1.3093, 1.4948, 1.6185, 1.9278, 2.2371, 2.8556],
        'mean': [98.5240, 98.3000, 98.2680, 98.2600, 98.3600, 98.0240, 98.1240]
    },
    2.0: {
        'sfp': [1.1477, 1.3693, 1.5908, 1.7385, 2.1078, 2.4771, 3.2156],
        'mean': [98.2080, 98.4800, 98.0200, 98.1160, 97.9060, 97.9780, 97.8280]
    },

}#heat map of original solution

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
    # 1.8: {
    #     'sfp': [1.1237, 1.3093, 1.4948, 1.6185, 1.9278, 2.2371, 2.8556],
    #     'mean': [98.5240, 98.3000, 98.2680, 98.2600, 98.3600, 98.0240, 98.1240]
    # },
    # 2.0: {
    #     'sfp': [1.1477, 1.3693, 1.5908, 1.7385, 2.1078, 2.4771, 3.2156],
    #     'mean': [98.2080, 98.4800, 98.0200, 98.1160, 97.9060, 97.9780, 97.8280]
    # },

}#heat map of compensated solution

overdrive_values5 = [' ', ' ',' ' ,' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
beta_values5 = ['/', '1.2', '1.05', '/', '/', '/', '/', '/', '/', '/', '/', '/', '/', '/']
xlabel4='Reference Voltage (V)'
ylabel4='Overdrive Ratio & $\\beta$ Value'
title4='Accuracy Mapping: $V_{ref}$ vs. Overdrive ($\\beta$) with $sf_p$ Values'






if __name__ == "__main__":
    #accuracy_samples_analysis(data3,group_metadata)
    #plot_weight_histogram(weights_path)
    #plot_parameter_fitting_grid(data5, overdrive_values5, beta_values5, xlabel4, ylabel4, title4)
    plot_twiny()

