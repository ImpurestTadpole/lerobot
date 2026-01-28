# 4-Channel RGBD Implementation Guide for LeRobot

This guide shows how to use 4-channel RGBD (RGB + Depth) data from RealSense cameras to train policies in LeRobot, following LeRobot's processor pipeline architecture.

## Overview

LeRobot policies use vision encoders that read `in_channels` from the image feature shape. For RGBD:
- **RGB images**: Shape `(3, H, W)` → `in_channels=3`
- **RGBD images**: Shape `(4, H, W)` → `in_channels=4`

## Quick Summary

This guide covers the complete RGBD integration workflow:

1. **Capture RGBD** in robot's `get_observation()` → Store as `(H, W, 4)` numpy array
2. **Create Dataset** → Automatically handles 4-channel images via `lerobot-record`
3. **Implement RGBD Processor** → Follows LeRobot's `ObservationProcessorStep` pattern
4. **Integrate Processor** → Add to `robot_observation_processor` pipeline
5. **Adapt Vision Encoder** → Policies automatically detect 4 channels and adapt
6. **Train Policy** → Use standard `lerobot-train` command

**Important: Camera Setup**
- **Head Camera**: RealSense D435i with RGBD (RGB + Depth) - **Only this camera has depth**
  - Provides workspace overview with 3D depth information
  - Output: 4-channel RGBD images `(H, W, 4)`
  - Used for navigation, obstacle avoidance, and 3D-aware manipulation
  
- **Wrist Cameras**: OpenCV cameras (RGB-only) - **No depth capability**
  - Provide close-up views for fine manipulation
  - Output: 3-channel RGB images `(H, W, 3)`
  - Lower latency, sufficient for close-range tasks

- **Visualization Optimization**: Depth images are automatically skipped in Rerun visualization for performance but still collected in dataset

**Key Concepts:**
- **Processor Pipeline**: RGBD processing happens in the observation processor pipeline
- **Feature Contracts**: Use `transform_features()` to declare shape transformations
- **Automatic Adaptation**: Policies automatically adapt vision encoders for 4 channels
- **Format Conversion**: Dataset stores `(H, W, 4)`, processor converts to `(C, H, W)` for policies

## Step 1: Capture RGBD Data During Recording

Modify your robot's `get_observation()` method to combine RGB and depth into a 4-channel array. **Important**: Only the head camera (RealSense) has depth capability. Wrist cameras are RGB-only.

### Camera Configuration

```python
# In config_xlerobot.py
cameras = {
    # Head camera: RealSense D435i with RGBD capability
    "head": RealSenseCameraConfig(
        serial_number_or_name="342222071125",
        fps=30,
        width=640,
        height=480,
        use_depth=True,  # ✅ Only head camera has depth
    ),
    
    # Wrist cameras: OpenCV cameras (RGB-only, no depth)
    "left_wrist": OpenCVCameraConfig(
        index_or_path="/dev/video8",
        fps=30,
        width=640,
        height=480,
        # No use_depth parameter - RGB-only camera
    ),
    "right_wrist": OpenCVCameraConfig(
        index_or_path="/dev/video6",
        fps=30,
        width=640,
        height=480,
        # No use_depth parameter - RGB-only camera
    ),
}
```

### Implementation in get_observation()

```python
# In xlerobot.py or your robot's get_observation() method
import numpy as np
import cv2

def get_observation(self) -> dict[str, Any]:
    """
    Get robot observation including RGBD images.
    
    **Camera Setup:**
    - Head camera: RealSense D435i (RGBD - RGB + Depth)
    - Wrist cameras: OpenCV cameras (RGB-only, no depth)
    
    Returns:
        Observation dictionary with:
        - RGBD images from head camera: (H, W, 4) format
        - RGB images from wrist cameras: (H, W, 3) format
    """
    obs = {}
    
    # ... get joint states, base velocities, etc. ...
    
    # Head camera: RealSense with RGBD capability
    if "head" in self.cameras and self.cameras["head"].use_depth:
        try:
            color_frame = self.cameras["head"].async_read()
            depth_frame = self.cameras["head"].async_read_depth()
            
            # Ensure both frames are same size
            if color_frame.shape[:2] != depth_frame.shape[:2]:
                depth_frame = cv2.resize(
                    depth_frame, 
                    (color_frame.shape[1], color_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST  # Preserve depth accuracy
                )
            
            # Normalize depth to [0, 255] range for storage
            # RealSense depth is typically uint16 in millimeters
            # Scale: depth_mm / 10.0 gives depth in cm, then normalize to [0, 255]
            # This assumes max depth of ~25.5 meters (255 * 0.1m)
            depth_normalized = np.clip(
                depth_frame.astype(np.float32) / 10.0,  # mm -> cm, then scale
                0, 255
            ).astype(np.uint8)
            
            # Stack RGB + Depth: (H, W, 3) + (H, W, 1) -> (H, W, 4)
            rgbd_image = np.dstack([color_frame, depth_normalized])
            
            # Store with proper observation key format
            # Use OBS_IMAGES namespace for consistency
            obs[f"observation.images.head_rgbd"] = rgbd_image  # Shape: (H, W, 4)
            
        except Exception as e:
            logger.warning(f"Failed to capture RGBD from head camera: {e}")
            # Fallback to RGB only
            try:
                color_frame = self.cameras["head"].async_read()
                obs["observation.images.head"] = color_frame
            except Exception:
                pass
    
    # Wrist cameras: RGB-only (no depth capability)
    # These are automatically captured as RGB images
    for cam_key in ["left_wrist", "right_wrist"]:
        if cam_key in self.cameras:
            try:
                rgb_frame = self.cameras[cam_key].async_read()
                obs[f"observation.images.{cam_key}"] = rgb_frame  # Shape: (H, W, 3)
            except Exception as e:
                logger.warning(f"Failed to capture from {cam_key} camera: {e}")
    
    return obs
```

**Important Notes:**
- **Only head camera** produces RGBD images (4 channels)
- **Wrist cameras** produce RGB images only (3 channels)
- Store RGBD images in the `observation.images.*` namespace for consistency
- Use `(H, W, 4)` format for RGBD, `(H, W, 3)` for RGB (dataset convention)
- Normalize depth to uint8 [0, 255] for efficient storage
- The RGBD processor will convert RGBD to `(C, H, W)` format for policies
- Depth images are automatically skipped in visualization for performance

## Step 2: Define Dataset Features with Mixed Camera Types

When creating your dataset, specify features for both RGBD (head) and RGB (wrist) cameras:

```python
# In your dataset creation or config
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_features = {
    # Head camera: RGBD (4 channels)
    "observation.images.head_rgbd": {
        "dtype": "video",  # or "image"
        "shape": (480, 640, 4),  # (H, W, C) format - RGBD
        "names": ["height", "width", "channels"],
    },
    # Wrist cameras: RGB-only (3 channels)
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),  # (H, W, C) format - RGB
        "names": ["height", "width", "channels"],
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),  # (H, W, C) format - RGB
        "names": ["height", "width", "channels"],
    },
    # ... other features (joint states, actions, etc.)
}

# Create dataset
dataset = LeRobotDataset.create(
    repo_id="your_username/xlerobot_rgbd",
    fps=30,
    robot_type="xlerobot",
    features=dataset_features,
    use_videos=True,
)
```

**Important**: LeRobot stores images as `(H, W, C)` but policies expect `(C, H, W)`. The conversion happens automatically during dataset loading.

## Step 3: Create RGBD Processor

Create a processor step following LeRobot's processor patterns. This processor handles RGBD preprocessing and integrates with the observation processor pipeline:

```python
# src/lerobot/processor/rgbd_processor.py
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import PolicyFeature, FeatureType, PipelineFeatureType
from lerobot.utils.constants import OBS_IMAGES


@dataclass
@ProcessorStepRegistry.register(name="rgbd_processor")
class RGBDProcessorStep(ObservationProcessorStep):
    """
    Process 4-channel RGBD images for policy input.
    
    This processor step handles depth normalization and RGB normalization separately,
    converting RGBD images from dataset format (H, W, 4) to policy format (C, H, W).
    
    **Processing Steps:**
    1. Splits RGBD into RGB (3 channels) and Depth (1 channel)
    2. Normalizes RGB: [0, 255] -> [0, 1]
    3. Converts depth from uint8 [0, 255] back to meters
    4. Normalizes depth using specified method (linear/log/inverse)
    5. Stacks back to 4-channel and converts to channel-first tensor
    
    **Attributes:**
        depth_scale: Scale factor to convert depth units (default: 10.0 for mm->m)
        depth_max: Maximum depth in meters for normalization (default: 10.0)
        normalize_rgb: Whether to normalize RGB to [0, 1] (default: True)
        normalize_depth: Whether to normalize depth (default: True)
        depth_normalization: Normalization method - "linear", "log", or "inverse" (default: "linear")
    """
    
    depth_scale: float = 10.0  # Convert mm to normalized range
    depth_max: float = 10.0  # Maximum depth in meters
    normalize_rgb: bool = True
    normalize_depth: bool = True
    depth_normalization: str = "linear"  # "linear", "log", "inverse"
    
    def _normalize_depth(self, depth: Tensor) -> Tensor:
        """Normalize depth channel using specified method.
        
        Args:
            depth: Depth tensor in meters
            
        Returns:
            Normalized depth tensor in [0, 1] range
        """
        if self.depth_normalization == "linear":
            # Linear scaling: [0, depth_max] -> [0, 1]
            return torch.clamp(depth / self.depth_max, 0.0, 1.0)
        elif self.depth_normalization == "log":
            # Logarithmic scaling (better for large depth ranges)
            return torch.log(depth + 1.0) / torch.log(torch.tensor(self.depth_max + 1.0, device=depth.device))
        elif self.depth_normalization == "inverse":
            # Inverse depth (common in vision)
            return 1.0 / (depth + 0.1)
        else:
            raise ValueError(f"Unknown depth normalization: {self.depth_normalization}")
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Process RGBD images in observation dictionary.
        
        This method is called by the ObservationProcessorStep base class
        and processes all observation keys containing RGBD data.
        
        Args:
            observation: Observation dictionary potentially containing RGBD images
            
        Returns:
            Processed observation dictionary with RGBD images converted to tensors
        """
        processed = observation.copy()
        
        for key, value in observation.items():
            # Look for RGBD images (keys containing "rgbd" or in images namespace)
            is_rgbd_key = (
                "rgbd" in key.lower() or 
                (key.startswith(f"{OBS_IMAGES}.") and isinstance(value, (np.ndarray, Tensor)))
            )
            
            if not is_rgbd_key:
                continue
            
            # Handle numpy arrays (from dataset) or tensors (from previous processors)
            if isinstance(value, np.ndarray):
                # From dataset: shape (H, W, 4) as uint8
                if value.ndim != 3 or value.shape[-1] != 4:
                    continue
                
                # Convert to tensor
                value = torch.from_numpy(value)
            
            if isinstance(value, Tensor):
                # Ensure channel-last format: (H, W, 4)
                if value.ndim == 4:  # (B, H, W, C) or (B, C, H, W)
                    if value.shape[1] == 4:  # Channel-first
                        value = value.permute(0, 2, 3, 1)  # (B, H, W, C)
                    # Remove batch dimension for processing
                    value = value[0]  # (H, W, C)
                elif value.ndim == 3 and value.shape[0] == 4:  # (C, H, W)
                    value = value.permute(1, 2, 0)  # (H, W, C)
                
                if value.shape[-1] != 4:
                    continue
                
                # Split RGB and Depth
                rgb = value[:, :, :3].float()  # (H, W, 3)
                depth = value[:, :, 3:4].float().squeeze(-1)  # (H, W)
                
                # Normalize RGB: [0, 255] -> [0, 1]
                if self.normalize_rgb:
                    rgb = rgb / 255.0
                
                # Convert depth from uint8 [0, 255] back to meters
                # Dataset stores depth as uint8 [0, 255] representing [0, depth_max] meters
                depth_meters = (depth / 255.0) * self.depth_max
                
                # Normalize depth
                if self.normalize_depth:
                    depth_normalized = self._normalize_depth(depth_meters)
                else:
                    depth_normalized = depth_meters / self.depth_max
                
                # Stack back: (H, W, 4)
                rgbd_processed = torch.cat([
                    rgb,
                    depth_normalized.unsqueeze(-1)
                ], dim=-1)  # (H, W, 4)
                
                # Convert to channel-first: (C, H, W)
                rgbd_tensor = rgbd_processed.permute(2, 0, 1).float()
                
                processed[key] = rgbd_tensor
        
        return processed
    
    def get_config(self) -> dict[str, Any]:
        """
        Returns JSON-serializable configuration for this processor step.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            "depth_scale": self.depth_scale,
            "depth_max": self.depth_max,
            "normalize_rgb": self.normalize_rgb,
            "normalize_depth": self.normalize_depth,
            "depth_normalization": self.depth_normalization,
        }
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Update feature shapes to reflect 4-channel RGBD input.
        
        This method declares how the processor transforms feature shapes,
        ensuring downstream components (policies) receive the correct input format.
        
        Args:
            features: Policy features dictionary with dataset feature shapes
            
        Returns:
            Updated features dictionary with 4-channel RGBD shapes
        """
        # Process observation features
        if PipelineFeatureType.OBSERVATION in features:
            obs_features = features[PipelineFeatureType.OBSERVATION]
            
            for key, feature in list(obs_features.items()):
                # Check if this is an RGBD feature
                is_rgbd = (
                    "rgbd" in key.lower() or
                    (key.startswith(f"{OBS_IMAGES}.") and feature.type == FeatureType.VISUAL)
                )
                
                if is_rgbd and len(feature.shape) == 3:
                    # Convert from dataset format (H, W, C) to policy format (C, H, W)
                    h, w, c = feature.shape
                    if c == 4:  # RGBD
                        # Update shape to channel-first: (4, H, W)
                        obs_features[key] = PolicyFeature(
                            type=feature.type,
                            shape=(4, h, w),
                        )
        
        return features
    
    def reset(self) -> None:
        """Reset processor state (no state to reset for this processor)."""
        pass
```

## Step 4: Adapt Vision Encoders for 4 Channels

Most LeRobot policies use ResNet backbones. You need to modify the first conv layer:

### Option A: Modify ACT Policy

```python
# src/lerobot/policies/act/modeling_act.py
# In ACT.__init__(), after creating backbone_model:

if self.config.image_features:
    # Get the first image feature to determine channels
    image_key = next(iter(self.config.image_features.keys()))
    in_channels = self.config.image_features[image_key].shape[0]
    
    backbone_model = getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        weights=config.pretrained_backbone_weights,
        norm_layer=FrozenBatchNorm2d,
    )
    
    # Adapt first conv layer for RGBD (4 channels)
    if in_channels == 4:
        first_conv = backbone_model.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        
        # Initialize: copy RGB weights, initialize depth channel
        with torch.no_grad():
            if first_conv.weight.shape[1] == 3:  # Original RGB
                # Copy RGB weights
                new_conv.weight[:, :3, :, :] = first_conv.weight
                # Initialize depth channel as average of RGB
                new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
            else:
                # Random initialization if not RGB
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            if first_conv.bias is not None:
                new_conv.bias = first_conv.bias
        
        backbone_model.conv1 = new_conv
    
    self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
```

### Option B: Create a Utility Function

```python
# src/lerobot/policies/utils/vision_utils.py
import torch
import torch.nn as nn

def adapt_vision_encoder_for_rgbd(backbone_model, in_channels=4):
    """
    Adapt a pretrained RGB vision encoder to accept RGBD (4-channel) input.
    
    Args:
        backbone_model: The vision backbone (e.g., ResNet)
        in_channels: Number of input channels (4 for RGBD)
    
    Returns:
        Modified backbone_model
    """
    if in_channels == 3:
        return backbone_model  # Already RGB
    
    # Get first conv layer
    if hasattr(backbone_model, 'conv1'):
        first_conv = backbone_model.conv1
    elif hasattr(backbone_model, 'stem') and hasattr(backbone_model.stem, 'conv'):
        first_conv = backbone_model.stem.conv
    else:
        raise ValueError("Could not find first conv layer in backbone")
    
    # Create new conv layer
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    
    # Initialize weights
    with torch.no_grad():
        if first_conv.weight.shape[1] == 3:  # Original RGB
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = first_conv.weight
            # Initialize depth channel
            # Option 1: Average of RGB channels
            new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
            # Option 2: Copy from red channel (often depth-like)
            # new_conv.weight[:, 3:4, :, :] = first_conv.weight[:, 0:1, :, :]
            # Option 3: Xavier initialization
            # nn.init.xavier_uniform_(new_conv.weight[:, 3:4, :, :])
        else:
            # Random initialization
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
    
    # Replace in model
    if hasattr(backbone_model, 'conv1'):
        backbone_model.conv1 = new_conv
    elif hasattr(backbone_model, 'stem'):
        backbone_model.stem.conv = new_conv
    
    return backbone_model
```

Then use it in your policy:

```python
# In ACT.__init__()
from lerobot.policies.utils.vision_utils import adapt_vision_encoder_for_rgbd

if self.config.image_features:
    image_key = next(iter(self.config.image_features.keys()))
    in_channels = self.config.image_features[image_key].shape[0]
    
    backbone_model = getattr(torchvision.models, config.vision_backbone)(
        weights=config.pretrained_backbone_weights,
        # ... other args
    )
    
    # Adapt for RGBD if needed
    if in_channels == 4:
        backbone_model = adapt_vision_encoder_for_rgbd(backbone_model, in_channels=4)
    
    self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
```

## Step 5: Update Policy Configuration

When training, ensure your config specifies 4-channel input:

```python
# In your training config or script
from lerobot.configs.train import TrainConfig
from lerobot.policies.act.configuration_act import ACTConfig

# Policy config
policy_config = ACTConfig(
    image_features={
        "observation.images.head_rgbd": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(4, 128, 128),  # 4 channels: RGBD
        ),
    },
    vision_backbone="resnet18",
    # ... other config
)

# Training config
train_config = TrainConfig(
    policy=policy_config,
    dataset_repo_id="your_username/xlerobot_rgbd",
    # ... other config
)
```

## Step 6: Integrate RGBD Processor into Pipeline

The RGBD processor should be added to the **robot observation processor pipeline** (not the policy processor). This ensures RGBD images are processed before being stored in the dataset and used by the policy.

### During Recording

The processor is automatically used when you define it in your robot's observation processor:

```python
# In your recording script or robot configuration
from lerobot.processor.factory import make_default_robot_observation_processor
from lerobot.processor.rgbd_processor import RGBDProcessorStep
from lerobot.processor.pipeline import RobotProcessorPipeline

# Create default observation processor
robot_observation_processor = make_default_robot_observation_processor()

# Add RGBD processor step
rgbd_step = RGBDProcessorStep(
    depth_scale=10.0,
    depth_max=10.0,
    normalize_rgb=True,
    normalize_depth=True,
    depth_normalization="linear",
)

# Insert RGBD processor into the pipeline
# It should run early to process images before other steps
robot_observation_processor.steps.insert(0, rgbd_step)
```

### During Training

For training, the RGBD processor should be part of the **preprocessor pipeline** that processes observations before they reach the policy:

```python
# Training script
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.rgbd_processor import RGBDProcessorStep

# Load dataset
dataset = LeRobotDataset("your_username/xlerobot_rgbd")

# Create policy
policy = make_policy(train_config.policy, ds_meta=dataset.meta)

# Create preprocessor and postprocessor pipelines
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=train_config.policy,
    pretrained_path=train_config.policy.pretrained_path,
    dataset_stats=dataset.meta.stats,
)

# Add RGBD processor to preprocessor if not already included
# Check if RGBD processor is needed
has_rgbd = any(
    "rgbd" in key.lower() 
    for key in train_config.policy.input_features.keys()
    if train_config.policy.input_features[key].type == FeatureType.VISUAL
)

if has_rgbd:
    rgbd_step = RGBDProcessorStep(
        depth_scale=10.0,
        depth_max=10.0,
        normalize_rgb=True,
        normalize_depth=True,
        depth_normalization="linear",
    )
    # Insert early in preprocessor pipeline
    preprocessor.steps.insert(0, rgbd_step)

# Train normally
# ... training loop
```

### Using Processor Overrides

You can also configure the RGBD processor via policy configuration overrides:

```python
# In your training config
from lerobot.configs.train import TrainConfig
from lerobot.policies.act.configuration_act import ACTConfig

policy_config = ACTConfig(
    image_features={
        "observation.images.head_rgbd": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(4, 128, 128),  # 4 channels: RGBD
        ),
    },
    # ... other config
)

# Processor overrides (applied when loading pretrained policies)
preprocessor_overrides = {
    "rgbd_processor": {
        "depth_max": 10.0,
        "depth_normalization": "linear",
    },
}
```

## Step 7: Feature Contract Validation

Before training, validate that your pipeline produces the expected features. This ensures compatibility between your dataset, processor, and policy:

```python
from lerobot.processor.pipeline import aggregate_pipeline_dataset_features
from lerobot.processor.core import create_initial_features, PipelineFeatureType
from lerobot.processor.rgbd_processor import RGBDProcessorStep

# Create RGBD processor
rgbd_processor = RGBDProcessorStep(
    depth_max=10.0,
    depth_normalization="linear",
)

# Define initial features (what your robot outputs)
initial_features = create_initial_features(
    observation={
        "observation.images.head_rgbd": (480, 640, 4),  # (H, W, C) from robot
    }
)

# Check how processor transforms features
transformed_features = rgbd_processor.transform_features({
    PipelineFeatureType.OBSERVATION: {
        "observation.images.head_rgbd": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(480, 640, 4),  # Dataset format
        ),
    }
})

# Expected output: shape should be (4, 480, 640) - channel-first
print(f"Transformed shape: {transformed_features[PipelineFeatureType.OBSERVATION]['observation.images.head_rgbd'].shape}")
# Output: (4, 480, 640)
```

## Step 8: Debugging Your RGBD Pipeline

Use LeRobot's debugging tools to verify your RGBD processor works correctly. See the [Debug Processor Pipeline guide](https://huggingface.co/docs/lerobot/debug_processor_pipeline) for detailed debugging techniques.

### Step-Through Debugging

Inspect intermediate results as data flows through your pipeline:

```python
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.core import create_transition, TransitionKey
from lerobot.processor.rgbd_processor import RGBDProcessorStep
import numpy as np

# Create a test observation with RGBD
test_observation = {
    "observation.images.head_rgbd": np.zeros((480, 640, 4), dtype=np.uint8),  # (H, W, 4)
}

# Create processor pipeline
rgbd_step = RGBDProcessorStep(depth_max=10.0)
pipeline = DataProcessorPipeline(steps=[rgbd_step])

# Step through the pipeline to inspect intermediate results
for step_idx, intermediate in enumerate(pipeline.step_through(test_observation)):
    print(f"After step {step_idx}:")
    obs = intermediate.get(TransitionKey.OBSERVATION)
    if obs and "head_rgbd" in obs:
        rgbd_tensor = obs["head_rgbd"]
        print(f"  Shape: {rgbd_tensor.shape}")  # Should be (4, 480, 640)
        print(f"  Dtype: {rgbd_tensor.dtype}")  # Should be float32
        print(f"  Range: [{rgbd_tensor.min():.3f}, {rgbd_tensor.max():.3f}]")  # Should be [0, 1]
```

### Using Hooks for Monitoring

Add hooks to monitor RGBD processing during runtime:

```python
def monitor_rgbd_processing(step_idx: int, transition):
    """Hook to monitor RGBD processing."""
    obs = transition.get(TransitionKey.OBSERVATION)
    if obs:
        for key, value in obs.items():
            if "rgbd" in key.lower() and isinstance(value, torch.Tensor):
                print(f"Step {step_idx}: {key} shape={value.shape}, dtype={value.dtype}")

# Register hook
pipeline.register_after_step_hook(monitor_rgbd_processing)

# Process data - hook will be called automatically
output = pipeline(test_observation)
```

## Step 9: Integration with Existing Processors

The RGBD processor should work alongside other observation processors. Here's the recommended order:

```python
from lerobot.processor.factory import make_default_robot_observation_processor
from lerobot.processor.rgbd_processor import RGBDProcessorStep
from lerobot.processor.rename_processor import RenameObservationsProcessorStep

# Create default pipeline
robot_observation_processor = make_default_robot_observation_processor()

# Add RGBD processor first (processes raw RGBD images)
rgbd_step = RGBDProcessorStep(depth_max=10.0)
robot_observation_processor.steps.insert(0, rgbd_step)

# Other processors (rename, normalize, etc.) will run after RGBD processing
# The pipeline order matters: RGBD -> Rename -> Normalize -> Device
```

### Processor Pipeline Order

The recommended order for observation processors:

1. **RGBD Processor** (first) - Converts RGBD format and normalizes depth
2. **Rename Processor** - Maps observation keys to policy keys
3. **Image Crop/Resize** - Resizes images to policy input size
4. **Normalizer** - Normalizes state/action features using dataset stats
5. **Device Processor** - Moves tensors to GPU/CPU

This order ensures RGBD images are processed before other transformations.

## Benefits of RGBD for XLeRobot

**Head Camera (RGBD)** provides:
1. **Better Depth Estimation**: Explicit depth vs. learning from RGB
2. **Fewer Training Episodes**: 20-40% reduction for 3D-aware tasks
3. **Improved Grasping**: Better object geometry understanding
4. **Navigation**: Enhanced obstacle avoidance

**Wrist Cameras (RGB-only)** provide:
1. **Close-up Views**: Detailed manipulation feedback
2. **Lower Latency**: No depth processing overhead
3. **Sufficient for Fine Manipulation**: RGB provides enough detail for close-range tasks

**Mixed Setup Benefits**:
- Head RGBD for workspace overview and navigation
- Wrist RGB for fine manipulation and grasp verification
- Best of both worlds: 3D awareness + low-latency close-up views

## Storage Considerations

RGBD datasets are larger:
- RGB: ~276 GB for 500 episodes (uncompressed)
- RGBD: ~460 GB for 500 episodes (uncompressed)
- With compression: RGBD ~69 GB vs RGB ~14 GB

## Complete Integration Example

Here's a complete example showing how to integrate RGBD end-to-end:

```python
# 1. Camera Configuration (config_xlerobot.py)
cameras = {
    # Head camera: RealSense D435i (RGBD)
    "head": RealSenseCameraConfig(
        serial_number_or_name="342222071125",
        fps=30,
        width=640,
        height=480,
        use_depth=True,  # ✅ Only head camera has depth
    ),
    # Wrist cameras: OpenCV (RGB-only)
    "left_wrist": OpenCVCameraConfig(...),   # RGB-only
    "right_wrist": OpenCVCameraConfig(...),  # RGB-only
}

# 2. Robot observation (xlerobot.py)
def get_observation(self):
    obs = {}
    # ... joint states, base velocities ...
    
    # Head camera: Capture RGBD (4 channels)
    if "head" in self.cameras and self.cameras["head"].use_depth:
        color = self.cameras["head"].async_read()
        depth = self.cameras["head"].async_read_depth()
        depth_norm = np.clip(depth.astype(np.float32) / 10.0, 0, 255).astype(np.uint8)
        rgbd = np.dstack([color, depth_norm])  # (H, W, 4)
        obs["observation.images.head_rgbd"] = rgbd
    
    # Wrist cameras: Capture RGB only (3 channels)
    for cam_key in ["left_wrist", "right_wrist"]:
        if cam_key in self.cameras:
            rgb = self.cameras[cam_key].async_read()  # RGB-only
            obs[f"observation.images.{cam_key}"] = rgb  # (H, W, 3)
    
    return obs

# 3. Dataset creation (automatic via lerobot-record)
# Dataset will have:
# - "observation.images.head_rgbd": (480, 640, 4) - RGBD from head camera
# - "observation.images.left_wrist": (480, 640, 3) - RGB from left wrist
# - "observation.images.right_wrist": (480, 640, 3) - RGB from right wrist

# 3. Processor pipeline setup
from lerobot.processor.factory import make_default_robot_observation_processor
from lerobot.processor.rgbd_processor import RGBDProcessorStep

robot_observation_processor = make_default_robot_observation_processor()
rgbd_step = RGBDProcessorStep(depth_max=10.0, depth_normalization="linear")
robot_observation_processor.steps.insert(0, rgbd_step)

# 4. Policy configuration
policy_config = ACTConfig(
    image_features={
        "observation.images.head_rgbd": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(4, 128, 128),  # After resizing in processor
        ),
    },
    vision_backbone="resnet18",
)

# 5. Vision encoder adaptation (automatic if shape[0] == 4)
# The policy will detect 4 channels and adapt the first conv layer

# 6. Training
# Use lerobot-train with your config - everything works automatically!
```

## Troubleshooting

### Issue: "Shape mismatch" errors
- **Cause**: Dataset stores (H, W, 4) but policy expects (4, H, W)
- **Fix**: Ensure `transform_features()` correctly converts shapes

### Issue: Depth values are all zeros
- **Cause**: Depth normalization or scaling incorrect
- **Fix**: Check `depth_scale` and `depth_max` match your camera's depth range

### Issue: Policy doesn't adapt to 4 channels
- **Cause**: Vision encoder adaptation not triggered
- **Fix**: Verify `image_features` shape is `(4, H, W)` not `(H, W, 4)`

### Issue: Processor not being called
- **Cause**: Processor not in the correct pipeline
- **Fix**: Add to `robot_observation_processor`, not `preprocessor`

## Best Practices

1. **Normalize depth consistently**: Use the same `depth_max` during recording and training
2. **Validate features**: Use `transform_features()` to verify shape transformations
3. **Test processor**: Use `step_through()` to debug processor behavior
4. **Monitor performance**: RGBD processing adds ~5-10ms overhead
5. **Storage optimization**: Consider depth compression if storage is limited

## Camera Setup Summary

**XLeRobot Configuration:**
- **Head Camera** (`head`): RealSense D435i
  - Type: RGBD (RGB + Depth)
  - Channels: 4 (R, G, B, D)
  - Use case: Workspace overview, navigation, 3D-aware tasks
  - Dataset key: `observation.images.head_rgbd` → Shape: `(480, 640, 4)`
  
- **Left Wrist Camera** (`left_wrist`): OpenCV camera
  - Type: RGB-only
  - Channels: 3 (R, G, B)
  - Use case: Close-up manipulation feedback
  - Dataset key: `observation.images.left_wrist` → Shape: `(480, 640, 3)`
  
- **Right Wrist Camera** (`right_wrist`): OpenCV camera
  - Type: RGB-only
  - Channels: 3 (R, G, B)
  - Use case: Close-up manipulation feedback
  - Dataset key: `observation.images.right_wrist` → Shape: `(480, 640, 3)`

**Key Points:**
- Only head camera produces RGBD images (4 channels)
- Wrist cameras produce RGB images only (3 channels)
- RGBD processor only processes head camera images
- Wrist camera images are handled by standard RGB processors
- Depth images are collected but not visualized (performance optimization)

## Next Steps

1. ✅ Implement RGBD capture in your robot's `get_observation()` (head camera only)
2. ✅ Create dataset with mixed features:
   - Head: 4-channel RGBD `(480, 640, 4)`
   - Wrist cameras: 3-channel RGB `(480, 640, 3)`
3. ✅ Add RGBD processor to observation pipeline (processes head camera)
4. ✅ Adapt vision encoder for 4 channels (automatic in policies for head camera)
5. ✅ Train and evaluate!

For questions or issues, check the LeRobot documentation:
- [Processor Guide](https://huggingface.co/docs/lerobot/processors_robots_teleop)
- [Implement Your Own Processor](https://huggingface.co/docs/lerobot/implement_your_own_processor)
- [Debug Processor Pipeline](https://huggingface.co/docs/lerobot/debug_processor_pipeline)

Or open an issue on [GitHub](https://github.com/huggingface/lerobot).
