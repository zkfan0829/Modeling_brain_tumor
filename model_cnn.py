
"""
RigidRegCNN: 3D CNN that ingests concatenated CT & MRI volumes and regresses 6 rigid parameters.

Primary goal:
- Accepts ANY input spatial size (B, 2, D, H, W). No hard asserts on (256,256,256).
- When D=H=W=256, the feature-map sizes follow the requested example.
- Uses AdaptiveAvgPool3d to produce a fixed 2048 features per sample regardless of input size.

Example (if input is 256^3):
 1) (B,  2, 256,256,256) --conv--> (B, 16, 128,128,128)
 2) (B, 16, 128,128,128) --conv--> (B, 32,  64, 64, 64)
 3) (B, 32,  64, 64, 64) --conv--> (B, 32,  32, 32, 32)
 4) (B, 32,  32, 32, 32) --conv--> (B, 32,  16, 16, 16)
 5) (B, 32,  16, 16, 16) --conv--> (B, 32,   8,  8,  8)

Then AdaptiveAvgPool3d -> (B, 32, 4, 4, 4), flatten -> 2048, FC -> 6 [rx, ry, rz, tx, ty, tz].
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RigidRegCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # --- Downsampling blocks (stride=2) ---
        self.conv1 = self._conv_block( 2, 16)  # (B,2,D,H,W)->(B,16,D/2,H/2,W/2)
        self.conv2 = self._conv_block(16, 32)  # -> (B,32,D/4,H/4,W/4)
        self.conv3 = self._conv_block(32, 32)  # -> (B,32,D/8,H/8,W/8)
        self.conv4 = self._conv_block(32, 32)  # -> (B,32,D/16,H/16,W/16)
        self.conv5 = self._conv_block(32, 32)  # -> (B,32,D/32,H/32,W/32)

        # Pool to get exact 2048 features per sample regardless of current D,H,W
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # -> (B,32,4,4,4)

        # Regression head
        self.fc = nn.Linear(32 * 4 * 4 * 4, 6)

        self._init_weights()

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 2, D, H, W) with arbitrary spatial size.
        Returns:
            params: Tensor (B, 6) = [rx, ry, rz, tx, ty, tz]
        """
        if x.ndim != 5:
            raise ValueError(f"Input must be 5D (B,C,D,H,W); got shape {tuple(x.shape)}")
        B, C, D, H, W = x.shape
        if C != 2:
            raise ValueError(f"Expected 2 channels (CT,MRI); got C={C}")

        # --- Feature extractor (shape comments assume D=H=W=256) ---
        x = self.conv1(x)  # (B,16,128,128,128) when 256^3 input
        x = self.conv2(x)  # (B,32, 64, 64, 64)
        x = self.conv3(x)  # (B,32, 32, 32, 32)
        x = self.conv4(x)  # (B,32, 16, 16, 16)
        x = self.conv5(x)  # (B,32,  8,  8,  8)

        # Normalize feature map size to 4x4x4 then flatten -> 2048
        x = self.pool(x)   # (B,32,4,4,4) for any input size
        x = torch.flatten(x, start_dim=1)  # (B, 2048)
        params = self.fc(x)  # (B,6)
        return params


if __name__ == "__main__":
    # Quick self-test for variable shapes
    for shp in [(2,2,256,256,256), (1,2,192,224,256)]:
        x = torch.randn(*shp)
        model = RigidRegCNN()
        with torch.no_grad():
            y = model(x)
        print(f"Input {shp} -> output {tuple(y.shape)}")
