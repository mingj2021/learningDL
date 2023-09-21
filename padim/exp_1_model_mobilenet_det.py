from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights, \
resnet101,ResNet101_Weights,  \
resnet50,ResNet50_Weights,   \
mobilenet_v3_large,MobileNet_V3_Large_Weights,   \
mobilenet_v2, MobileNet_V2_Weights,swin_v2_t,Swin_V2_T_Weights,swin_v2_s,Swin_V2_S_Weights,   \
vgg16, VGG16_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights

from random import sample
from anomaly_map import AnomalyMapGenerator
from multi_variate_gaussian import MultiVariateGaussian
from pathlib import Path
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.stats import norm
from torch.distributions import Normal
import cv2 as cv
import matplotlib.pyplot as plt
from torchsummary import summary
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class FeatureExtractorMobilenet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v2(weights=MobileNet_V2_Weights)
        self.block0 = model.features[0:2]
        self.block1 = model.features[2:4]
        self.block2 = model.features[4:7]
        self.block3 = model.features[7:14]
        # self.block4 = model.features[14:17]

    def forward(self, x):
        features_0 = self.block0(x)
        features_1 = self.block1(features_0)
        features_2 = self.block2(features_1)
        features_3 = self.block3(features_2)
        # features_4 = self.block4(features_3)
        features = {}
        features['0'] = features_0
        features['1'] = features_1
        features['2'] = features_2
        features['3'] = features_3
        # features['4'] = features_4
        return features

class PadimModel(nn.Module):
    def __init__(self,
                 input_size: tuple[int, int],
                 backbone: str, 
                 layers: list[str]
                 ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractorMobilenet_v2()
        
        self.n_features_original, self.n_patches = self._deduce_dim(input_size)
        print('---------------------- n_features_original=', self.n_features_original)
        self.idx = torch.arange(0,self.n_features_original,step=3, dtype=torch.long)
        print('---------------------- shape=', self.idx.shape)
        print('---------------------- max=', self.idx.max())
        
        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=input_size, sigma=4)

        self.gaussian = MultiVariateGaussian(self.idx.shape[0], self.n_patches)

        # minMax metric
        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def _deduce_dim(self, input_size: tuple[int, int]) -> tuple[int, int]:
        dryrun_input = torch.empty(1, 3, *input_size)
        features = self.feature_extractor(dryrun_input)
        dimensions_mapping = {
            layer: {"num_features": features[layer].shape[1],
                    "resolution": features[layer].shape[2:]}
            for layer in self.layers
        }

        first_layer_resolution = dimensions_mapping[self.layers[0]]["resolution"]
        n_patches = torch.tensor(first_layer_resolution).prod().int().item()

        n_features_original = sum(
            dimensions_mapping[layer]["num_features"] for layer in self.layers)  # type: ignore

        return n_features_original, n_patches

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(
                layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)
            embeddings = self.generate_embedding(features)

        # if self.training:
        #     output = embeddings
        self.max = torch.max(self.max, torch.max(embeddings))
        self.min = torch.min(self.min, torch.min(embeddings))

        return embeddings


def normalize_min_max(
    targets: np.ndarray | np.float32 | Tensor,
    threshold: float | np.ndarray | Tensor,
    min_val: float | np.ndarray | Tensor,
    max_val: float | np.ndarray | Tensor,
) -> np.ndarray | Tensor:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(
            1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(
            0))  # pylint: disable=not-callable
    else:
        raise ValueError(
            f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized


def standardize(
    targets: np.ndarray | Tensor,
    mean: float | np.ndarray | Tensor,
    std: float | np.ndarray | Tensor,
    center_at: float | None = None,
) -> np.ndarray | Tensor:
    """Standardize the targets to the z-domain."""
    if isinstance(targets, np.ndarray):
        targets = np.log(targets)
    elif isinstance(targets, Tensor):
        targets = torch.log(targets)
    else:
        raise ValueError(
            f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    standardized = (targets - mean) / std
    if center_at:
        standardized -= (center_at - mean) / std
    return standardized


def normalize_cdf(targets: np.ndarray | Tensor, threshold: float | np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Normalize the targets by using the cumulative density function."""
    if isinstance(targets, Tensor):
        return normalize_torch(targets, threshold)
    if isinstance(targets, np.ndarray):
        return normalize_numpy(targets, threshold)
    raise ValueError(
        f"Targets must be either Tensor or Numpy array. Received {type(targets)}")


def normalize_torch(targets: Tensor, threshold: Tensor) -> Tensor:
    """Normalize the targets by using the cumulative density function, PyTorch version."""
    device = targets.device
    image_threshold = threshold.cpu()

    dist = Normal(torch.Tensor([0]), torch.Tensor([1]))
    normalized = dist.cdf(targets.cpu() - image_threshold).to(device)
    return normalized


def normalize_numpy(targets: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Normalize the targets by using the cumulative density function, Numpy version."""
    return norm.cdf(targets - threshold)


def _normalize(
    pred_scores: Tensor | np.float32,
    metadata: dict,
    anomaly_maps: Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray | Tensor | None, float]:
    """Applies normalization and resizes the image.

    Args:
        pred_scores (Tensor | np.float32): Predicted anomaly score
        metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
            additional meta data such as image shape. This variable comprises such info.
        anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

    Returns:
        tuple[np.ndarray | Tensor | None, float]: Post processed predictions that are ready to be
            visualized and predicted scores.
    """

    # min max normalization
    if "min" in metadata and "max" in metadata:
        if anomaly_maps is not None:
            anomaly_maps = normalize_min_max(
                anomaly_maps,
                metadata["pixel_threshold"],
                metadata["min"],
                metadata["max"],
            )
        pred_scores = normalize_min_max(
            pred_scores,
            metadata["image_threshold"],
            metadata["min"],
            metadata["max"],
        )

    # standardize pixel scores
    if "pixel_mean" in metadata.keys() and "pixel_std" in metadata.keys():
        if anomaly_maps is not None:
            anomaly_maps = standardize(
                anomaly_maps, metadata["pixel_mean"], metadata["pixel_std"], center_at=metadata["image_mean"]
            )
            anomaly_maps = normalize_cdf(
                anomaly_maps, metadata["pixel_threshold"])

    # standardize image scores
    if "image_mean" in metadata.keys() and "image_std" in metadata.keys():
        pred_scores = standardize(
            pred_scores, metadata["image_mean"], metadata["image_std"])
        pred_scores = normalize_cdf(pred_scores, metadata["image_threshold"])

    return anomaly_maps, float(pred_scores)


def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(
            f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(B * H * W, device=image.device,
                       dtype=image.dtype).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)


def connected_components_gpu(image: Tensor, num_iterations: int = 1000) -> Tensor:
    """Perform connected component labeling on GPU and remap the labels from 0 to N.

    Args:
        image (Tensor): Binary input image from which we want to extract connected components (Bx1xHxW)
        num_iterations (int): Number of iterations used in the connected component computation.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = connected_components(image, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: Tensor) -> Tensor:
    """Connected component labeling on CPU.

    Args:
        image (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = torch.zeros_like(image)
    label_idx = 1
    for i, mask in enumerate(image):
        mask = mask.squeeze().numpy().astype(np.uint8)
        _, comps = cv.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            components[i, 0, ...][np.where(comps == label)] = label_idx
            label_idx += 1
    return components.int()


def masks_to_boxes(masks: Tensor, anomaly_maps: Tensor | None = None) -> tuple[list[Tensor], list[Tensor]]:
    """Convert a batch of segmentation masks to bounding box coordinates.

    Args:
        masks (Tensor): Input tensor of shape (B, 1, H, W), (B, H, W) or (H, W) 
        anomaly_maps (Tensor | None, optional): Anomaly maps of shape (B, 1, H, W), (B, H, W) or (H, W) which are
            used to determine an anomaly score for the converted bounding boxes.

    Returns:
        list[Tensor]: A list of length B where each element is a tensor of shape (N, 4) containing the bounding box
            coordinates of the objects in the masks in xyxy format.
        list[Tensor]: A list of length B where each element is a tensor of length (N) containing an anomaly score for
            each of the converted boxes.
    """
    height, width = masks.shape[-2:]
    # reshape to (B, 1, H, W) and cast to float
    masks = masks.view((-1, 1, height, width)).float()
    if anomaly_maps is not None:
        anomaly_maps = anomaly_maps.view((-1,) + masks.shape[-2:])

    if masks.is_cuda:
        batch_comps = connected_components_gpu(masks).squeeze(1)
    else:
        batch_comps = connected_components_cpu(masks).squeeze(1)

    batch_boxes = []
    batch_scores = []
    for im_idx, im_comps in enumerate(batch_comps):
        labels = torch.unique(im_comps)
        im_boxes = []
        im_scores = []
        for label in labels[labels != 0]:
            y_loc, x_loc = torch.where(im_comps == label)
            # add box
            box = Tensor([torch.min(x_loc), torch.min(y_loc), torch.max(
                x_loc), torch.max(y_loc)]).to(masks.device)
            im_boxes.append(box)
            if anomaly_maps is not None:
                im_scores.append(torch.max(anomaly_maps[im_idx, y_loc, x_loc]))
        batch_boxes.append(torch.stack(im_boxes) if im_boxes else torch.empty(
            (0, 4), device=masks.device))
        batch_scores.append(torch.stack(
            im_scores) if im_scores else torch.empty(0, device=masks.device))

    return batch_boxes, batch_scores

def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv.applyColorMap(anomaly_map, cv.COLORMAP_JET)
    anomaly_map = cv.cvtColor(anomaly_map, cv.COLOR_BGR2RGB)
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max


    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    superimposed_map = cv.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map




def train(category = 'bottle'):
    input_size=(512, 512)
    model = PadimModel(input_size=input_size, backbone='resnet18', layers=['0','1','2', '3'])# layers=['layer1','layer2','layer3']
    # summary(model.to('cuda:0'),(3,256,256),depth=3)
    ROOT = Path(
        'C:/Users/77274/workspace/projects/anomalib/datasets/MVTec/{}/train/good'.format(category))
    files = os.listdir(str(ROOT))
    embeddings = []
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor()])
    for f in files:
        im = Image.open(str(ROOT / f))
        im = im.convert('RGB')
        x = transforms(im).unsqueeze(0)
        embedding = model.forward(x)

        embeddings.append(embedding)
        print('embedding shape : ', embedding.shape)

    embeddings = torch.vstack(embeddings)
    [mean, inv_covariance] = model.gaussian.fit(embeddings)
    torch.save({'mean':mean,'inv_covariance':inv_covariance,'min': model.min,'max': model.max},'{}.pt'.format(category))
    # stats = torch.load('dist.pt')
    # mean, inv_covariance = stats['mean'],stats['inv_covariance']


    im = Image.open(
        'C:/Users/77274/workspace/projects/anomalib/datasets/MVTec/wafer/test/good/000009.png')
    im = im.convert('RGB')
    
    x = transforms(im).unsqueeze(0)
    embedding = model.forward(x)
    predictions = model.anomaly_map_generator(
        embedding=embedding, mean=mean, inv_covariance=inv_covariance)
    anomaly_map = predictions.detach()
    # tmp = (anomaly_map -anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    # tmp = tmp.numpy() * 255
    # tmp = tmp.astype(np.uint8).squeeze()
    # ret,thresh1 = cv.threshold(tmp,175,255,cv.THRESH_BINARY)
    # cv.imshow('',thresh1)
    # cv.waitKey()

    
    print('---------------',model.min,' ', model.max)
    pred_mask = anomaly_map >= 14
    pred_boxes = masks_to_boxes(pred_mask)[0][0].numpy()

    img = np.array(im)
    img = cv.resize(img, input_size)
    for box in pred_boxes:
        cv.rectangle(img, (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])), (255, 0, 0))

    pred_mask = pred_mask.squeeze().numpy().astype(np.uint8) * 255
    cv.imshow('pred_mask', pred_mask)
    cv.imshow('img', img)
    cv.waitKey()

def predict(category = 'bottle'):
    stats = torch.load('{}.pt'.format(category))
    mean = stats['mean']
    inv_covariance = stats['inv_covariance']
    input_size = (512, 512)
    model = PadimModel(input_size=input_size, backbone='resnet18', layers=['0','1','2', '3'])#
    # model = PadimModel(input_size=input_size, backbone='resnet18', layers=['features_1'])# layers=['layer1','layer2','layer3']
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor()])
    ROOT = 'C:/Users/77274/workspace/projects/anomalib/datasets/MVTec/{}/test/fabric_border/'.format(category)
    for f in os.listdir(ROOT):
        im = Image.open(ROOT + f)
        gt_f = f.split('.')[0] + '_mask.' + f.split('.')[1]
        im = im.convert('RGB')
        gt = (ROOT + gt_f).replace('test','ground_truth')
        gt = Image.open(gt)
        gt = np.array(gt)
        x = transforms(im).unsqueeze(0)
        embedding = model.forward(x)
        predictions = model.anomaly_map_generator(
            embedding=embedding, mean=mean, inv_covariance=inv_covariance)
        anomaly_map = predictions.detach()
        print(anomaly_map.min(), anomaly_map.max())
        # thres = anomaly_map.max() - (anomaly_map.max() - anomaly_map.min()) / 3
        # thres = thres.item()
        thres = 11
        pred_mask = anomaly_map >= thres
        pred_boxes = masks_to_boxes(pred_mask)[0][0].numpy()

        img = np.array(im)
        img = cv.resize(img, input_size)
        img_ = img.copy()
        for box in pred_boxes:
            cv.rectangle(img, (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), (255, 0, 0))

        pred_mask = pred_mask.squeeze().numpy().astype(np.uint8) * 255

        anomaly_map = predictions.detach().cpu().numpy()
        anomaly_map = anomaly_map.squeeze()
        pred_score = anomaly_map.reshape(-1).max()
        metadata = {'image_threshold':0.,'pixel_threshold':thres,'min':stats['min'].item(),'max':stats['max'].item()}
        anomaly_map, pred_score = _normalize(anomaly_maps=anomaly_map, pred_scores=pred_score, metadata=metadata)
        heat_map = superimpose_anomaly_map(anomaly_map, img_, normalize=False)
        figure, axis = plt.subplots(1, 4, figsize=(8,8))
        axis[0].imshow(heat_map)
        axis[1].imshow(gt)
        axis[2].imshow(pred_mask,cmap='gray')
        axis[3].imshow(img)
        plt.show()
    
    # print()



if __name__ == '__main__':
    # train('zipper')
    predict('zipper')
    # from torchsummary import summary
    # model = vgg16(weights=VGG16_Weights.DEFAULT).to("cuda:0")
    # print(model)
    # # model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1).to("cuda:0")
    # # model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).to("cuda:0")
    # input = torch.randn(1, 3, 256, 256).to("cuda:0")
    # x = model(input)
    # summary(model, input_size=(3, 256, 256))
