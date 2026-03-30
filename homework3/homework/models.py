from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        self.features = nn.Sequential(            
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

        # TODO: implement
        #pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        input_mean = self.input_mean.to(x.device)
        input_std = self.input_std.to(x.device)
        z = (x - input_mean[None, :, None, None]) / input_std[None, :, None, None]    
        #z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        z = self.features(z)
        logits = self.classifier(z)
        # TODO: replace with actual forward pass
        #logits = torch.randn(x.size(0), 6)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    """High-capacity U-Net with skip-conv fusion"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        # Normalization
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))
        
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        # Decoder + conv fusion
        self.up4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                   nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        self.up1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        # Heads
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.depth_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        z = (x - self.input_mean[None,:,None,None]) / self.input_std[None,:,None,None]

        # Encoder
        d1 = self.down1(z)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Decoder with skip convs
        u4 = self.up4(d4)
        u4 = torch.cat([u4, d3], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = self.conv1(u1)

        if u1.shape[-2:] != (H, W):
            u1 = F.interpolate(u1, size=(H, W), mode="bilinear", align_corners=False)

        logits = self.seg_head(u1)
        depth = torch.sigmoid(self.depth_head(u1)).squeeze(1)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
