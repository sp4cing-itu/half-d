import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================
# HALVIT STAGE FOR RESNEXT - Shares single projection matrix bidirectionally
# =====================================================================================
class ResNeXtHalvitStage(nn.Module):
    """
    Implements parameter sharing for ResNeXt blocks within a stage.
    Following the true halvit principle: single bidirectional projection matrix.
    Maintains cardinality (groups) for the 3x3 grouped convolution.
    """
    expansion = 4
    
    def __init__(self, in_channels, mid_channels, num_blocks, stride=1, 
                 groups=32, width_per_group=8):
        super().__init__()
        
        out_channels = mid_channels * self.expansion
        # Calculate the actual width for grouped convolution (ResNeXt formula)
        width = int(mid_channels * (width_per_group / 64.0)) * groups
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        
        # ========================================================================
        # HALVIT PRINCIPLE: SINGLE SHARED PROJECTION MATRIX
        # Used bidirectionally like in ResNet50 halvit
        # ========================================================================
        # Single projection matrix: width <-> out_channels
        self.convp = nn.Parameter(torch.empty((width, out_channels, 1, 1)))
        nn.init.kaiming_normal_(self.convp, mode='fan_out', nonlinearity='relu')
        
        # ========================================================================
        # CHANNEL ADAPTATION FOR FIRST BLOCK
        # ========================================================================
        # Need to handle in_channels != out_channels for first block
        self.channel_adapter = None
        if in_channels != out_channels:
            # Simple 1x1 conv to adapt channels for first block
            self.channel_adapter = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # ========================================================================
        # SEPARATE COMPONENTS FOR EACH BLOCK
        # ========================================================================
        self.bn_pre_list = nn.ModuleList()  # BN before projection
        self.conv2_list = nn.ModuleList()   # 3x3 grouped conv (not shared)
        self.bn2_list = nn.ModuleList()     # BN after 3x3 conv
        self.bn_post_list = nn.ModuleList() # BN after back-projection
        
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            
            # Pre-projection BatchNorm
            self.bn_pre_list.append(nn.BatchNorm2d(out_channels))
            
            # 3x3 grouped convolution (ResNeXt cardinality preserved)
            self.conv2_list.append(
                nn.Conv2d(width, width, kernel_size=3, stride=block_stride,
                         padding=1, groups=groups, bias=False)
            )
            
            # BatchNorm after 3x3 conv
            self.bn2_list.append(nn.BatchNorm2d(width))
            
            # Post-projection BatchNorm
            self.bn_post_list.append(nn.BatchNorm2d(out_channels))
        
        # ========================================================================
        # DOWNSAMPLE PATH
        # ========================================================================
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.num_blocks = num_blocks
        
    def forward(self, x):
        # Handle channel adaptation for first block if needed
        if self.channel_adapter is not None:
            x_adapted = self.channel_adapter(x)
        else:
            x_adapted = x
        
        out = x_adapted
        
        for i in range(self.num_blocks):
            if i == 0:
                identity = x
                # Downsample if needed (only for first block)
                if self.downsample is not None:
                    identity = self.downsample(x)
            else:
                identity = out
            
            # Pre-projection BN and ReLU
            block_out = self.bn_pre_list[i](out)
            block_out = self.relu(block_out)
            
            # Forward projection: out_channels -> width (using convp)
            block_out = F.conv2d(block_out, self.convp)
            
            # 3x3 grouped convolution (maintains cardinality)
            block_out = self.conv2_list[i](block_out)
            block_out = self.bn2_list[i](block_out)
            block_out = self.relu(block_out)
            
            # Backward projection: width -> out_channels (using transposed convp)
            block_out = F.conv2d(block_out, self.convp.permute(1, 0, 2, 3))
            block_out = self.bn_post_list[i](block_out)
            
            # Residual connection
            out = block_out + identity
            out = self.relu(out)
        
        return out


# =====================================================================================
# STANDARD RESNEXT BOTTLENECK (for comparison and first two stages)
# =====================================================================================
class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, 
                 groups=32, width_per_group=8):
        super().__init__()
        
        width = int(mid_channels * (width_per_group / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, mid_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        return out


# =====================================================================================
# RESNEXT-101 WITH HALVIT PARAMETER SHARING
# =====================================================================================
class ResNeXt101Halvit(nn.Module):
    def __init__(self, num_classes=1000, groups=32, width_per_group=8,
                 use_halvit_from_stage=3, init_cfg=None, **kwargs):
        """
        ResNeXt-101 with Halvit parameter sharing.
        
        Args:
            num_classes: Number of output classes
            groups: Cardinality (number of groups)
            width_per_group: Width of each group
            use_halvit_from_stage: Start using halvit from this stage (1-indexed)
                                  Default is 3 (apply to layer3 and layer4)
            init_cfg: Initialization config dict (for MMDetection compatibility)
            **kwargs: Additional arguments for MMDetection compatibility
        """
        super().__init__()
        
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.use_halvit_from_stage = use_halvit_from_stage
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNeXt-101 configuration: [3, 4, 23, 3]
        self.layer1 = self._make_layer(64, blocks=3, stride=1, stage=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2, stage=2)
        self.layer3 = self._make_layer(256, blocks=23, stride=2, stage=3)
        self.layer4 = self._make_layer(512, blocks=3, stride=2, stage=4)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, mid_channels, blocks, stride, stage):
        """
        Create a layer with either standard ResNeXt blocks or Halvit shared weights.
        """
        use_halvit = stage >= self.use_halvit_from_stage
        
        if use_halvit:
            # Use Halvit parameter sharing for this stage
            out_channels = mid_channels * 4
            layer = ResNeXtHalvitStage(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                num_blocks=blocks,
                stride=stride,
                groups=self.groups,
                width_per_group=self.width_per_group
            )
            self.in_channels = out_channels
            return layer
        else:
            # Use standard ResNeXt blocks
            layers = []
            out_channels = mid_channels * 4
            
            # Downsample if needed
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsample = None
            
            # First block
            layers.append(ResNeXtBottleneck(
                self.in_channels, mid_channels, stride, downsample,
                self.groups, self.width_per_group
            ))
            self.in_channels = out_channels
            
            # Remaining blocks
            for _ in range(1, blocks):
                layers.append(ResNeXtBottleneck(
                    self.in_channels, mid_channels, groups=self.groups,
                    width_per_group=self.width_per_group
                ))
            
            return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        return tuple(outs)


# =====================================================================================
# PARAMETER COUNTING AND TESTING
# =====================================================================================
def count_parameters_detailed(model):
    """Count parameters with detailed breakdown."""
    total_params = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        
        # Identify shared parameters (convp in Halvit stages)
        if 'convp' in name:
            shared_params += params
            print(f"Shared weight: {name:50s} - {params:,} params")
    
    print(f"\nTotal shared parameters: {shared_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    return total_params


if __name__ == "__main__":
    print("="*80)
    print("ResNeXt-101-32x8d with Halvit Parameter Sharing")
    print("="*80)
    
    # Create model with halvit applied to layer3 and layer4
    model_halvit = ResNeXt101Halvit(
        num_classes=1000, 
        groups=32, 
        width_per_group=8,
        use_halvit_from_stage=3  # Apply halvit from stage 3 onwards
    )
    
    print("\nModel with Halvit (applied to layer3 and layer4):")
    halvit_params = count_parameters_detailed(model_halvit)
    
    # Compare with original
    print("\n" + "="*80)
    print("Comparison with Original ResNeXt-101-32x8d:")
    print("="*80)
    
    from torchvision.models import resnext101_32x8d
    original_model = resnext101_32x8d(weights=None)
    original_params = sum(p.numel() for p in original_model.parameters())
    
    print(f"Original ResNeXt-101-32x8d parameters: {original_params:,}")
    print(f"Halvit ResNeXt-101-32x8d parameters:   {halvit_params:,}")
    print(f"Parameter reduction: {original_params - halvit_params:,} "
          f"({(1 - halvit_params/original_params)*100:.2f}% reduction)")
    
    # Test forward pass
    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)

    # You can also test with halvit applied to all stages
    print("\n" + "="*80)
    print("Testing with Halvit applied to ALL stages:")
    print("="*80)
    
    model_full_halvit = ResNeXt101Halvit(
        num_classes=10,
        groups=32,
        width_per_group=8,
        use_halvit_from_stage=1  # Apply halvit from stage 1 (all stages)
    )
    
    full_halvit_params = sum(p.numel() for p in model_full_halvit.parameters())
    print(f"Full Halvit parameters: {full_halvit_params:,}")
    print(f"Maximum parameter reduction: {original_params - full_halvit_params:,} "
          f"({(1 - full_halvit_params/original_params)*100:.2f}% reduction)")
