from __future__ import annotations
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def gaussian(window_size: int, sigma: float) -> Tensor:
    device, dtype = None, None
    if isinstance(sigma, Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-(x**2.0) / (2 * sigma**2))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool = False) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma: gaussian standard deviation in the x and y
         direction.
        force_even: overrides requirement for odd kernel size.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def compute_kernel_size(sigma_val: float) -> int:
    """Compute kernel size from sigma value.

    Args:
        sigma_val (float): Sigma value.

    Returns:
        int: Kernel size.
    """
    return 2 * int(4.0 * sigma_val + 0.5) + 1


class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d.

    Makes use of kornia functions, but most notably the kernel is not computed
    during the forward pass, and does not depend on the input size. As a caveat,
    the number of channels that are expected have to be provided during initialization.
    """

    def __init__(
        self,
        sigma: float | tuple[float, float],
        channels: int = 1,
        kernel_size: int | tuple[int, int] | None = None,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        """Initialize model, setup kernel etc..

        Args:
            sigma (float | tuple[float, float]): standard deviation to use for constructing the Gaussian kernel.
            channels (int): channels of the input. Defaults to 1.
            kernel_size (int | tuple[int, int] | None): size of the Gaussian kernel to use. Defaults to None.
            normalize (bool, optional): Whether to normalize the kernel or not (i.e. all elements sum to 1).
                Defaults to True.
            border_type (str, optional): Border type to use for padding of the input. Defaults to "reflect".
            padding (str, optional): Type of padding to apply. Defaults to "same".
        """
        super().__init__()
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.channels = channels

        if kernel_size is None:
            kernel_size = (compute_kernel_size(sigma[0]), compute_kernel_size(sigma[1]))
        else:
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.kernel: Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)
        self.kernel.unsqueeze_(0).unsqueeze_(0)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Blur the input with the computed Gaussian.

        Args:
            input_tensor (Tensor): Input tensor to be blurred.

        Returns:
            Tensor: Blurred output tensor.
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (tuple): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(self, image_size: tuple, sigma: int = 4) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    def compute_distance(embedding: Tensor, stats: list[Tensor]) -> Tensor:
        """Compute anomaly score to the patch in position(i,j) of a test image.

        Ref: Equation (2), Section III-C of the paper.

        Args:
            embedding (Tensor): Embedding Vector
            stats (list[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        distances = distances.clamp(0).sqrt()

        return distances

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """Apply gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (Tensor): Anomaly score for the test image(s).

        Returns:
            Filtered anomaly scores
        """

        blurred_anomaly_map = self.blur(anomaly_map)
        return blurred_anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor) -> Tensor:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and inv_covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            inv_covariance (Tensor): Inverse Covariance matrix of the multivariate gaussian distribution.

        Returns:
            Output anomaly score.
        """

        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map

    def forward(self, **kwargs) -> Tensor:
        """Returns anomaly_map.

        Expects `embedding`, `mean` and `covariance` keywords to be passed explicitly.

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(embedding=embedding, mean=mean, covariance=covariance)

        Raises:
            ValueError: `embedding`. `mean` or `covariance` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            raise ValueError(f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}")

        embedding: Tensor = kwargs["embedding"]
        mean: Tensor = kwargs["mean"]
        inv_covariance: Tensor = kwargs["inv_covariance"]

        return self.compute_anomaly_map(embedding, mean, inv_covariance)
