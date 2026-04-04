import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,  F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,  F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetGenerator(nn.Module):
    """
    Encoder channel sizes  (after each enc block):
      e1: 64   e2: 128   e3: 256   e4: 512   bottleneck: 512

    Decoder — up block output, then cat with skip:
      d4 = up4(b)   512 → 256  then cat e4(512)  → 768  ... no, keep it simple:

    We use a simpler, battle-tested channel scheme below.
    """

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each block: Conv2d stride=2 → halve spatial, defined output channels
        self.enc1 = self._enc(in_channels,  features,     norm=False)  # →[64,  128,128]
        self.enc2 = self._enc(features,     features * 2)               # →[128,  64, 64]
        self.enc3 = self._enc(features * 2, features * 4)               # →[256,  32, 32]
        self.enc4 = self._enc(features * 4, features * 8)               # →[512,  16, 16]

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = self._enc(features * 8, features * 8, norm=False)  # →[512, 8, 8]

        # ── Decoder ──────────────────────────────────────────────────────────
        # up4: takes bottleneck(512) → outputs 512, then cat with e4(512) = 1024
        self.up4   = self._dec(features * 8,      features * 8)   # 512  → 512
        self.att4  = AttentionGate(features * 8,  features * 8,  features * 4)
        # after cat: 512+512 = 1024

        # up3: takes 1024 → outputs 256, then cat with e3(256) = 512
        self.up3   = self._dec(features * 8 * 2,  features * 4)   # 1024 → 256
        self.att3  = AttentionGate(features * 4,  features * 4,  features * 2)
        # after cat: 256+256 = 512

        # up2: takes 512 → outputs 128, then cat with e2(128) = 256
        self.up2   = self._dec(features * 4 * 2,  features * 2)   # 512  → 128
        self.att2  = AttentionGate(features * 2,  features * 2,  features)
        # after cat: 128+128 = 256

        # up1: takes 256 → outputs 64, then cat with e1(64) = 128
        self.up1   = self._dec(features * 2 * 2,  features)       # 256  → 64
        self.att1  = AttentionGate(features,       features,      features // 2)
        # after cat: 64+64 = 128

        # ── Final layer ───────────────────────────────────────────────────────
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _enc(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4,
                            stride=2, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _dec(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ── Encode ───────────────────────────────────────────────────────────
        e1 = self.enc1(x)          # [B,  64, 128, 128]
        e2 = self.enc2(e1)         # [B, 128,  64,  64]
        e3 = self.enc3(e2)         # [B, 256,  32,  32]
        e4 = self.enc4(e3)         # [B, 512,  16,  16]
        b  = self.bottleneck(e4)   # [B, 512,   8,   8]

        # ── Decode with attention-gated skips ────────────────────────────────
        d4 = self.up4(b)                     # [B, 512, 16, 16]
        e4 = self.att4(d4, e4)               # att: g=512, x=512 → [B, 512, 16, 16]
        d4 = torch.cat([d4, e4], dim=1)      # [B, 1024, 16, 16]

        d3 = self.up3(d4)                    # [B, 256, 32, 32]
        e3 = self.att3(d3, e3)               # att: g=256, x=256 → [B, 256, 32, 32]
        d3 = torch.cat([d3, e3], dim=1)      # [B, 512, 32, 32]

        d2 = self.up2(d3)                    # [B, 128, 64, 64]
        e2 = self.att2(d2, e2)               # att: g=128, x=128 → [B, 128, 64, 64]
        d2 = torch.cat([d2, e2], dim=1)      # [B, 256, 64, 64]

        d1 = self.up1(d2)                    # [B,  64, 128, 128]
        e1 = self.att1(d1, e1)               # att: g=64,  x=64  → [B,  64, 128, 128]
        d1 = torch.cat([d1, e1], dim=1)      # [B, 128, 128, 128]

        return self.final(d1)                # [B,   3, 256, 256]


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,  features,     4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features,     features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))