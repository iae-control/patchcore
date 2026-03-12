"""PatchCore model - v4: ONNX Runtime + TensorRT EP for fast feature extraction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
from typing import List, Tuple, Optional
from pathlib import Path
from src import config


class FeatureExtractor(nn.Module):
    """Extract multi-scale features from WideResNet50.
    
    v4: Optional ONNX Runtime + TensorRT acceleration.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)
        
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.to(device)
        
        if config.USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
            print(f"  Using DataParallel: {torch.cuda.device_count()} GPUs")
            self.layer1 = nn.DataParallel(self.layer1)
            self.layer2 = nn.DataParallel(self.layer2)
            self.layer3 = nn.DataParallel(self.layer3)
        
        self.eval()
        
        # Use torch.compile for GPU acceleration
        try:
            self._compiled = torch.compile(self._forward_impl, mode="max-autotune")
            with torch.no_grad():
                dummy = torch.randn(2, 3, 256, 256, device=self.device)
                self._compiled(dummy)
            print("  torch.compile: max-autotune enabled")
            self._use_compiled = True
        except Exception as e:
            print(f"  torch.compile failed ({e}), using eager mode")
            self._use_compiled = False

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward pass."""
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract layer2+layer3 features. AMP FP16 + torch.compile."""
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if self._use_compiled:
                out = self._compiled(x)
            else:
                out = self._forward_impl(x)
        return out.float()  # back to FP32 for coreset


def extract_features(dataset, extractor: FeatureExtractor, desc: str = "Extracting") -> np.ndarray:
    """Extract features for entire dataset. Returns (N, D) array."""
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    all_features = []
    for batch_tiles, _ in tqdm(loader, desc=desc, leave=False, 
                                unit="batch", dynamic_ncols=True):
        feats = extractor(batch_tiles)
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def greedy_coreset_selection(
    features: np.ndarray,
    ratio: float = None,
    projection_dim: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy coreset selection with GPU-accelerated distance computation."""
    ratio = ratio or config.CORESET_RATIO
    projection_dim = projection_dim or config.CORESET_PROJECTION_DIM
    n, d = features.shape
    target_count = max(1, int(n * ratio))

    if target_count >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj_matrix = rng.randn(d, projection_dim).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    proj_t = torch.from_numpy(proj_matrix).to(device)
    feat_t = torch.from_numpy(features.astype(np.float32)).to(device)
    projected_t = feat_t @ proj_t

    selected = [rng.randint(n)]
    min_distances = torch.full((n,), float('inf'), device=device)

    for _ in tqdm(range(target_count - 1), desc="Coreset selection", 
                  leave=False, unit="pt", dynamic_ncols=True):
        last = projected_t[selected[-1]]
        dists = torch.sum((projected_t - last) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, dists)
        next_idx = torch.argmax(min_distances).item()
        selected.append(next_idx)

    indices = np.array(selected)
    return features[indices], indices


class PatchCoreModel:
    """PatchCore model: holds coreset memory bank and scores anomalies."""

    def __init__(self):
        self.memory_bank: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray):
        self.memory_bank, _ = greedy_coreset_selection(features)
        print(f"  Memory bank: {self.memory_bank.shape[0]} / {features.shape[0]} features")

    def score(self, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        assert self.memory_bank is not None, "Model not fitted"
        n = features.shape[0]
        scores = np.zeros(n, dtype=np.float32)
        bank = torch.from_numpy(self.memory_bank).cuda()

        for i in tqdm(range(0, n, batch_size), desc="Scoring", 
                      leave=False, unit="batch", dynamic_ncols=True):
            batch = torch.from_numpy(features[i:i + batch_size]).cuda()
            dists = torch.cdist(batch.unsqueeze(0), bank.unsqueeze(0)).squeeze(0)
            min_dists, _ = dists.min(dim=1)
            scores[i:i + batch_size] = min_dists.cpu().numpy()

        return scores

    def save(self, path):
        np.save(path, self.memory_bank)

    def load(self, path):
        self.memory_bank = np.load(path, allow_pickle=True)
