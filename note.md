## CNN Concepts

### Input and Output Channels
- **Input Channels**: The number of channels in the input feature map (e.g., 3 for RGB images: red, green, blue).
- **Output Channels**: The number of filters applied, determining the depth of the output feature map. Each filter produces one output channel.

### Kernel/Filter and Channels
- **Kernel/Filter**: A small matrix (e.g., 3x3) that slides over the input to compute convolutions. It learns patterns like edges or textures.
- **Channels in Kernel**: Kernels have the same number of channels as the input. For example, a 3x3 kernel for RGB input has 3 channels (one per input channel). The output is summed across channels.

### Kernel Size and Stride
- **Kernel Size**: The dimensions of the kernel (e.g., 3x3, 5x5). Smaller sizes (like 3x3) are common for efficiency and capturing local features.
- **Stride**: The step size the kernel moves across the input. Stride 1 moves one pixel at a time; higher strides (e.g., 2) reduce output size and computation.

### Deciding Numbers in Channels
- **Input Channels**: Fixed by data (e.g., 1 for grayscale, 3 for RGB).
- **Output Channels**: Chosen based on model complexity. Start with small numbers (e.g., 32, 64) and increase in deeper layers. Use heuristics like powers of 2, or tune via experimentation for task performance (e.g., more channels for complex datasets).

### 3x3 Convolution
- A convolution using a 3x3 kernel. It's popular because:
  - Captures local patterns effectively.
  - Reduces parameters compared to larger kernels.
  - Can be stacked to achieve larger receptive fields (e.g., two 3x3 layers mimic one 5x5).
  - Computationally efficient, especially with strides and padding.