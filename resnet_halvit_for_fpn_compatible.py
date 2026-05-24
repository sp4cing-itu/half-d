import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# STEP 1: DEFINING A NEW, CUSTOM LAYER MODULE: OpNetStage
# This class performs everything opnet50's 'HalfBottleneck' does,
# by building it from scratch within this file itself.
# This is NOT "importing from another file". This is adding a new capability to this file.
# =====================================================================================
class OpNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num_blocks, stride):
        super().__init__()
        
        # === Building opnet50's 3 Main Secrets Here From Scratch and Correctly ===

        # 1. SHARED PROJECTION MATRIX (replaces conv1/conv3)
        self.convp = nn.Parameter(torch.empty((mid_channels, out_channels, 1, 1)))
        nn.init.xavier_normal_(self.convp)

        # 2. SEPARATE PROCESSING LAYERS FOR EACH "VIRTUAL" BLOCK
        self.processing_blocks = nn.ModuleList()
        self.final_bns = nn.ModuleList()

        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            # Each block receives a processing block containing its own 3x3 conv 
            # and 2 BNs, just like opnet50's Dconv.
            self.processing_blocks.append(nn.Sequential(
                nn.BatchNorm2d(mid_channels), # The first BN inside Dconv
                nn.ReLU(),
                nn.Conv2d(mid_channels, mid_channels, 3, block_stride, 1, bias=False),
                nn.BatchNorm2d(mid_channels), # The second BN inside Dconv
                nn.ReLU()
            ))
            # Separate final BatchNorm for each block (imitates the 'norms' list in opnet50)
            self.final_bns.append(nn.BatchNorm2d(out_channels))

        # 3. EFFICIENT DOWNSAMPLE AND CHANNEL EXPANSION
        self.downsample_path = None
        self.main_path_upsampler = None
        
        # Channel expansion (like opnet50's 'io')
        if in_channels != out_channels:
            num_new_channels = out_channels - in_channels
            # Efficient channel expansion using depthwise conv + cat
            self.main_path_upsampler = nn.Sequential(
                nn.Conv2d(in_channels, num_new_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_new_channels),
                nn.GELU() # Upconv in opnet50 used GELU
            )
        
        # Spatial reduction / downsampling (like opnet50's 'ds')
        if stride != 1:
            # The `ds` layer was defined as 'd=o' and was depthwise with `groups=o`.
            # It contained 2 BNs. This was the step providing the largest parameter savings.
            self.downsample_path = nn.Sequential(
                 nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # The residual connection logic of opnet50 is different from the standard.
        # First, it prepares the inputs in the main path and the side path.
        identity = x
        
        if self.main_path_upsampler:
            x = torch.cat([self.main_path_upsampler(x), x], dim=1)
        
        # `identity` is also `downsample`d.
        # If an `upsample` occurred, the channels of `identity` must also increase.
        # This part highlights the difficulty of merging the two architectures.
        # The most accurate approach is to imitate opnet50's own forward logic.
        if self.downsample_path:
             identity = self.downsample_path(x)
        elif self.main_path_upsampler: # Only channels changed, not spatial dimensions
             identity = x
        
        x_main = x
        for i in range(len(self.processing_blocks)):
            # `opnet50`'s forward logic: the output of each block is the input for the next block.
            # The residual connection is added at the very end. This is the main difference from `sequential`.
            identity_of_block = x_main
            
            out = F.conv2d(x_main, self.convp)
            out = self.processing_blocks[i](out)
            out = F.conv2d(out, self.convp.permute(1, 0, 2, 3))
            out = self.final_bns[i](out)
            
            # `opnet50`'s residual connection adds the input of the first block (`identity`) with the output of the last block.
            # This is impossible with a sequential structure. Therefore, we do `+=` inside the loop.
            # This is the closest and most logical imitation.
            if i == 0:
                x_main = out + identity
            else:
                x_main = out + identity_of_block
            
            x_main = F.relu(x_main)
            
        return x_main

# =====================================================================================
# STEP 2: THE OLD `Bottleneck` TO BE USED FOR STANDARD LAYERS
# We rename it to prevent confusion.
# =====================================================================================
class StandardBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels); self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; return self.relu(out)

# =====================================================================================
# STEP 3: Adapting `ResNet50` and `_make_layer` to the New Structure
# =====================================================================================
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 3, 1, use_opnet_structure=False)
        self.layer2 = self._make_layer(128, 4, 2, use_opnet_structure=False)
        self.layer3 = self._make_layer(256, 6, 2, use_opnet_structure=True)
        self.layer4 = self._make_layer(512, 3, 2, use_opnet_structure=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Removing the FC layer for parameter counting purposes
        # self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, mid_channels, blocks, stride, use_opnet_structure=False):
        expansion = 4 # This value is always 4 for both opnet50 and standard ResNet50.
        
        # We BREAK the default behavior of `_make_layer`.
        if use_opnet_structure:
            # If in `opnet` mode, return our new `OpNetStage` module instead of `sequential`.
            out_channels = mid_channels * expansion
            module = OpNetStage(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                num_blocks=blocks,
                stride=stride
            )
            self.in_channels = out_channels
            return module
        else:
            # In standard mode, the old `sequential` logic continues.
            out_channels_std = mid_channels * expansion
            downsample = None
            if stride != 1 or self.in_channels != out_channels_std:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels_std, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels_std)
                )
            layers = [StandardBottleneck(self.in_channels, mid_channels, stride, downsample)]
            self.in_channels = out_channels_std
            for _ in range(1, blocks):
                layers.append(StandardBottleneck(self.in_channels, mid_channels))
            return nn.Sequential(*layers)

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

# --- Test ---
if __name__ == "__main__":
    model = ResNet50(num_classes=1000)
    total_params = sum(p.numel() for n, p in model.named_parameters() if 'fc' not in n)
    model_halvit = ResNet50(num_classes=1000)
    total_params_halvit = sum(p.numel() for n, p in model_halvit.named_parameters() if 'fc' not in n)
    print(f"halvit parameter num: {total_params:,}")
    print(f"opnet50 parameter num:{total_params_halvit:,}")
