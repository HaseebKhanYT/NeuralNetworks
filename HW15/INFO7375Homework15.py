import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal


class SeparableConvolution:
    def __init__(self):
        pass

    def add_padding(self, image: np.ndarray, padding: int) -> np.ndarray:
        if len(image.shape) == 2:
            return np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        else:
            return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    def depthwise_convolution(self,
                              image: np.ndarray,
                              kernel: np.ndarray,
                              stride: int = 1,
                              padding: int = 0) -> np.ndarray:
        """
        Perform depthwise convolution.
        Each channel is convolved with its own kernel independently.

        Args:
            image: Input image of shape (H, W, C)
            kernel: Depthwise kernel of shape (K_H, K_W, C) or (K_H, K_W)
            stride: Stride for convolution
            padding: Padding to add to the image

        Returns:
            Output of depthwise convolution with same number of channels as input
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if len(kernel.shape) == 2:
            kernel = kernel[:, :, np.newaxis]

        # Add padding if specified
        if padding > 0:
            image = self.add_padding(image, padding)

        H, W, C = image.shape
        K_H, K_W, K_C = kernel.shape

        # For depthwise conv, number of kernels should match number of channels
        assert C == K_C, f"Number of channels in image ({C}) must match kernel channels ({K_C})"

        # Calculate output dimensions
        out_H = (H - K_H) // stride + 1
        out_W = (W - K_W) // stride + 1

        # Initialize output
        output = np.zeros((out_H, out_W, C))

        # Perform depthwise convolution
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * stride
                    h_end = h_start + K_H
                    w_start = j * stride
                    w_end = w_start + K_W

                    # Extract receptive field for current channel
                    receptive_field = image[h_start:h_end, w_start:w_end, c]

                    # Convolve with corresponding kernel
                    output[i, j, c] = np.sum(receptive_field * kernel[:, :, c])

        return output

    def pointwise_convolution(self,
                              image: np.ndarray,
                              kernel: np.ndarray,
                              stride: int = 1) -> np.ndarray:
        """
        Perform pointwise (1x1) convolution.
        Combines information across channels using 1x1 kernels.

        Args:
            image: Input image of shape (H, W, C)
            kernel: Pointwise kernel of shape (1, 1, C, N) or (C, N)
                   where C is input channels and N is output channels
            stride: Stride for convolution (usually 1 for pointwise)

        Returns:
            Output of pointwise convolution with N channels
        """
        # Handle grayscale images
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        H, W, C = image.shape

        # Reshape kernel if needed
        if len(kernel.shape) == 2:
            # Assume shape is (C, N)
            kernel = kernel.reshape(1, 1, C, -1)
        elif len(kernel.shape) == 3:
            # Assume shape is (1, 1, C*N) and needs reshaping
            kernel = kernel.reshape(1, 1, C, -1)

        _, _, K_C, N = kernel.shape

        assert C == K_C, f"Number of channels in image ({C}) must match kernel input channels ({K_C})"

        # Calculate output dimensions
        out_H = H // stride
        out_W = W // stride

        # Initialize output
        output = np.zeros((out_H, out_W, N))

        # Perform pointwise convolution
        for n in range(N):
            for i in range(out_H):
                for j in range(out_W):
                    h_idx = i * stride
                    w_idx = j * stride

                    # Pointwise conv: sum across all channels at a single spatial location
                    output[i, j, n] = np.sum(
                        image[h_idx, w_idx, :] * kernel[0, 0, :, n])

        return output

    def convolve(self,
                 image: np.ndarray,
                 kernel: np.ndarray,
                 conv_type: Literal['depthwise', 'pointwise'] = 'depthwise',
                 stride: int = 1,
                 padding: int = 0) -> np.ndarray:
        """
        Main convolution function that performs either depthwise or pointwise convolution.

        Args:
            image: Input image
            kernel: Convolution kernel
            conv_type: Type of convolution ('depthwise' or 'pointwise')
            stride: Stride for convolution
            padding: Padding for convolution

        Returns:
            Convoluted image
        """
        if conv_type == 'depthwise':
            return self.depthwise_convolution(image, kernel, stride, padding)
        elif conv_type == 'pointwise':
            return self.pointwise_convolution(image, kernel, stride)
        else:
            raise ValueError(
                f"conv_type must be 'depthwise' or 'pointwise', got {conv_type}")

    def visualize_results(self,
                          original: np.ndarray,
                          result: np.ndarray,
                          conv_type: str,
                          kernel: Optional[np.ndarray] = None):
        """
        Visualize the original and convoluted images.

        Args:
            original: Original input image
            result: Convolution result
            conv_type: Type of convolution performed
            kernel: Kernel used (optional, for display)
        """
        fig, axes = plt.subplots(1, 3 if kernel is not None else 2,
                                 figsize=(12 if kernel is not None else 10, 4))

        # Original image
        if len(original.shape) == 3 and original.shape[2] == 3:
            axes[0].imshow(original)
        else:
            axes[0].imshow(original.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Kernel visualization (if provided)
        if kernel is not None:
            if len(kernel.shape) >= 2:
                # For depthwise, show first channel; for pointwise, reshape and show
                if conv_type == 'depthwise':
                    kernel_vis = kernel[:, :, 0] if len(
                        kernel.shape) > 2 else kernel
                else:
                    kernel_vis = kernel.squeeze()
                    if len(kernel_vis.shape) > 2:
                        kernel_vis = kernel_vis.reshape(-1,
                                                        kernel_vis.shape[-1])

                im = axes[1].imshow(kernel_vis, cmap='coolwarm')
                axes[1].set_title(f'{conv_type.capitalize()} Kernel')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Result
        result_idx = 2 if kernel is not None else 1
        if len(result.shape) == 3:
            if result.shape[2] == 3:
                axes[result_idx].imshow(result)
            else:
                # Show first channel for multi-channel output
                axes[result_idx].imshow(result[:, :, 0], cmap='gray')
                axes[result_idx].set_title(
                    f'{conv_type.capitalize()} Result (Channel 0)')
        else:
            axes[result_idx].imshow(result.squeeze(), cmap='gray')
            axes[result_idx].set_title(f'{conv_type.capitalize()} Result')
        axes[result_idx].axis('off')

        plt.tight_layout()
        plt.show()


def create_sample_image(size: Tuple[int, int] = (28, 28), channels: int = 3) -> np.ndarray:
    """Create a sample image for testing."""
    np.random.seed(42)
    if channels == 1:
        # Create a simple pattern for grayscale
        image = np.zeros(size)
        image[size[0]//4:3*size[0]//4, size[1]//4:3*size[1]//4] = 1
        image[size[0]//3:2*size[0]//3, size[1]//3:2*size[1]//3] = 0.5
        return image
    else:
        # Create RGB image with different patterns per channel
        image = np.zeros((*size, channels))
        for c in range(channels):
            image[:, :, c] = np.random.rand(*size) * 0.5
            # Add some structure
            if c == 0:  # Red channel - horizontal gradient
                image[:, :, c] += np.linspace(0, 0.5, size[1]).reshape(1, -1)
            elif c == 1:  # Green channel - vertical gradient
                image[:, :, c] += np.linspace(0, 0.5, size[0]).reshape(-1, 1)
            else:  # Blue channel - diagonal pattern
                image[:, :, c] += np.eye(size[0], size[1]) * 0.5
        return np.clip(image, 0, 1)


def create_depthwise_kernel(kernel_size: Tuple[int, int] = (3, 3), channels: int = 3) -> np.ndarray:
    """Create a sample depthwise kernel."""
    kernel = np.zeros((*kernel_size, channels))

    # Different filter for each channel
    # Channel 0: Edge detection (horizontal)
    kernel[:, :, 0] = np.array([[-1, -1, -1],
                                [2,  2,  2],
                                [-1, -1, -1]])[:kernel_size[0], :kernel_size[1]]

    if channels > 1:
        # Channel 1: Edge detection (vertical)
        kernel[:, :, 1] = np.array([[-1,  2, -1],
                                    [-1,  2, -1],
                                    [-1,  2, -1]])[:kernel_size[0], :kernel_size[1]]

    if channels > 2:
        # Channel 2: Blur
        kernel[:, :, 2] = np.ones(kernel_size) / \
            (kernel_size[0] * kernel_size[1])

    return kernel


def create_pointwise_kernel(input_channels: int = 3, output_channels: int = 2) -> np.ndarray:
    """Create a sample pointwise (1x1) kernel."""
    # Random weights for channel mixing
    np.random.seed(42)
    kernel = np.random.randn(1, 1, input_channels, output_channels) * 0.5
    return kernel


# Main execution
if __name__ == "__main__":
    # Initialize the convolution class
    conv = SeparableConvolution()

    # Create sample data
    image = create_sample_image(size=(28, 28), channels=3)

    # Example 1: Depthwise Convolution
    print("=" * 50)
    print("DEPTHWISE CONVOLUTION")
    print("=" * 50)

    depthwise_kernel = create_depthwise_kernel(kernel_size=(3, 3), channels=3)

    # Perform depthwise convolution with padding
    result_depthwise = conv.convolve(
        image=image,
        kernel=depthwise_kernel,
        conv_type='depthwise',
        stride=1,
        padding=1
    )

    print(f"Input shape: {image.shape}")
    print(f"Depthwise kernel shape: {depthwise_kernel.shape}")
    print(f"Output shape: {result_depthwise.shape}")
    print(f"Note: Output channels = Input channels = {image.shape[2]}")

    # Visualize
    conv.visualize_results(image, result_depthwise,
                           'depthwise', depthwise_kernel)

    # Example 2: Pointwise Convolution
    print("\n" + "=" * 50)
    print("POINTWISE CONVOLUTION")
    print("=" * 50)

    pointwise_kernel = create_pointwise_kernel(
        input_channels=3, output_channels=2)

    # Perform pointwise convolution
    result_pointwise = conv.convolve(
        image=image,
        kernel=pointwise_kernel,
        conv_type='pointwise',
        stride=1
    )

    print(f"Input shape: {image.shape}")
    print(f"Pointwise kernel shape: {pointwise_kernel.shape}")
    print(f"Output shape: {result_pointwise.shape}")
    print(f"Note: Output channels = {pointwise_kernel.shape[-1]}")

    # Visualize
    conv.visualize_results(image, result_pointwise,
                           'pointwise', pointwise_kernel)

    # Example 3: Complete Depthwise Separable Convolution
    print("\n" + "=" * 50)
    print("DEPTHWISE SEPARABLE CONVOLUTION (Combined)")
    print("=" * 50)

    # Step 1: Depthwise
    intermediate = conv.convolve(
        image=image,
        kernel=depthwise_kernel,
        conv_type='depthwise',
        stride=1,
        padding=1
    )

    # Step 2: Pointwise
    final_output = conv.convolve(
        image=intermediate,
        kernel=pointwise_kernel,
        conv_type='pointwise',
        stride=1
    )

    print(f"Original shape: {image.shape}")
    print(f"After depthwise: {intermediate.shape}")
    print(f"After pointwise: {final_output.shape}")
    print(f"\nParameter reduction compared to standard convolution:")
    std_params = 3 * 3 * 3 * 2  # Standard 3x3 conv: K_H * K_W * C_in * C_out
    sep_params = (3 * 3 * 3) + (1 * 1 * 3 * 2)  # Depthwise + Pointwise
    print(f"Standard conv parameters: {std_params}")
    print(f"Separable conv parameters: {sep_params}")
    print(f"Reduction: {(1 - sep_params/std_params)*100:.1f}%")
