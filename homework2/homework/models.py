"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return F.cross_entropy(logits, target)
        #raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        self.linear = nn.Linear(3 * h * w, num_classes)
        #raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)  # flatten to (B, 3*64*64)
        logits = self.linear(x_flat)  # (B,6)
        return logits
        #raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        input_dim = 3 * h * w
        hidden_dim = 128  
        self.model = nn.Sequential(
            nn.Flatten(),           
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        #raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
        #raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim = 128,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()   

        input_dim = 3 * h * w
        layers=[]
        #for i in range(num_layers - 1):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.model = nn.Sequential(*layers)

        #raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
        #raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = 3 * h * w

        # First layer maps input_dim -> hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Residual hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Flatten input
        self.flatten = nn.Flatten()

        #raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        x = self.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            # Residual connection: add input to layer output
            x = self.relu(layer(x) + x)

        x = self.output_layer(x)
        return x
        #raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
