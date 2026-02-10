# Unsupervised-learning-in-sparsely-activated-convolutional-spiking-neural-networks-enabled-by-VDSP

Unofficial PyTorch Implementation: Unsupervised CSNN with VDSP
This repository contains a PyTorch-based refactoring and reproduction of the paper:

"Unsupervised and efficient learning in sparsely activated convolutional spiking neural networks enabled by voltage-dependent synaptic plasticity"


Goupy et al., Neuromorphic Computing and Engineering (2023) 

This implementation focuses on reproducing the Voltage-Dependent Synaptic Plasticity (VDSP) rule in a Convolutional Spiking Neural Network (CSNN) using Single-Spike Integrate-and-Fire (SSIF) neurons, experimenting different architectures for various contexts.

Visualization of learned SNN weights (Binary Features) after training on MNIST.

🚀 Key Features & Modifications
This project is not a direct translation; it includes several architectural experimental changes to test performance and usability in PyTorch:

1. Refactored Reproduction
This codebase faithfully reproduces the algorithm described in Goupy et al. (2023), specifically the unsupervised learning phase using the hardware-friendly VDSP rule and the readout layer using a Linear SVM.

2. Layer-wise vs. Step-wise Processing
Unlike many SNN implementations that process the entire network one time-step at a time (Step-wise), this implementation utilizes Layer-wise processing ( although it is not necessarily more efficient or hardware friendly).

The entire temporal dimension (timesteps) is calculated for one layer before passing the full spike train to the next layer.

This approach possibly leverages PyTorch's vectorization capabilities for faster training and inference( if the training process could be adapted to multiple batch size).

3. Native PyTorch & CUDA Support
Unified Data Structure: All data, including potentials and spikes, are handled as torch.Tensor objects.

GPU Acceleration: The code includes robust device management. Simply toggle device='cuda' to accelerate the unsupervised training and feature extraction on your GPU( however, training with GPU is slower in the current state, given the network still needs to be extended to process batch size bigger than 1).

4. Standardized CSNN Layers
The code defines modular classes for the Spiking Convolutional Layer and Spiking Pooling Layer.

Standardized I/O: Both layers accept and return a standard tuple: (potential_tensor, spike_tensor).

Robust Pooling: The pooling layer correctly handles the "Single-Spike" constraint, ensuring that the refractory period logic is preserved alongside the max-pooling operation.

5. Visualization Tools
Includes a dedicated Visualize.py script to generate:


Weight Grids: Visualizes the learned convolutional kernels, demonstrating the "binary weight" phenomenon described in the paper.

Prediction Demos: Runs inference on random test samples and displays the input image alongside the predicted label.

📂 Project Structure
Layers.py: Contains the CsnnLayer (with VDSP logic) and SnnPooling classes.


Model.py: Assembles the CSNN and the Linear SVM readout  into a unified model class. Handles feature extraction and flattening.

Training.py: Main script for the unsupervised training loop.

Visualize.py: Loads trained checkpoints to visualize weights and perform inference.

🛠️ Usage
Prerequisites
Python 3.x

PyTorch( 2.10.0+cu130)

scikit-learn (for the SVM readout)

Matplotlib (for visualization)

1. Training the SNN
Run the training script to learn the convolutional kernels via VDSP.

Bash
python Training.py

The script will automatically download MNIST, apply TTFS encoding, and train the network. Checkpoints are saved to the checkpoints/ directory.

2. Visualization & Prediction
Once trained, use the visualization script to inspect the learned features and test the model.

Bash
python Visualize.py
📊 Results
As described in the original paper, the VDSP rule causes the weights to converge towards binary values (0 or 1).

Learned Weights (This Implementation): The visualization below shows the 7x7 kernels after training. The clear black-and-white patterns indicate that the network has successfully learned to extract edge and shape features from the digit dataset using the modified LTD/LTP rules.

📝 Citation
If you use this code, please cite the original paper:

Code snippet
@article{goupy2023unsupervised,
  title={Unsupervised and efficient learning in sparsely activated convolutional spiking neural networks enabled by voltage-dependent synaptic plasticity},
  author={Goupy, Gaspard and Juneau-Fecteau, Alexandre and Garg, Nikhil and Balafrej, Ismael and Alibart, Fabien and Frechette, Luc and Drouin, Dominique and Beilliard, Yann},
  journal={Neuromorphic Computing and Engineering},
  volume={3},
  number={1},
  pages={014001},
  year={2023},
  publisher={IOP Publishing}
}
