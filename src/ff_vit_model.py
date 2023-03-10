import math

import torch
import torch.nn as nn

from src import utils

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.fc = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                # Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # FeedForward(dim, mlp_dim, dropout = dropout)
            ])

    def forward(self, x):        
        for operation in self.fc:
            x = operation(x) + x
            x = operation(x) + x

        self.x = x

        return x

class FF_ViT_model(torch.nn.Module):
    """The ViT model trained with Forward-Forward (FF)."""

    def __init__(self, opt, emb_dropout = 0.):
        super(FF_ViT_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.transformer.dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad()

        # # Initialize the model.
        image_height, image_width = pair(opt.input.image_size)
        patch_height, patch_width = pair(opt.transformer.patch_size)

        dims = [opt.transformer.dim] * opt.model.num_layers
        heads = [opt.transformer.heads] * opt.model.num_layers
        mlp_dim = [opt.model.hidden_dim] * opt.model.num_layers
        dim_head = [opt.transformer.dim_head] * opt.model.num_layers
        channels = opt.input.image_channels

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dims[0] - opt.input.num_classes),
            # nn.LayerNorm(dims[0] - opt.input.num_classes),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dims[0]))
        self.dropout = nn.Dropout(emb_dropout)

        self.model = nn.ModuleList([TransformerBlock(dims[0], heads[0], dim_head[0], mlp_dim[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(TransformerBlock(dims[i], heads[i], dim_head[i], mlp_dim[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i] * self.num_patches, device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] * self.num_patches for i in range(self.opt.model.num_layers - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, self.opt.input.num_classes, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        # self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        
        x = z[:,:-self.opt.input.num_classes]
        y = z[:,-self.opt.input.num_classes:]
        y = torch.unsqueeze(y, 1)
        y = repeat(y, 'b () d -> b p d', p=self.num_patches)
        x = rearrange(x, 'b (c h w) -> b c h w', c=self.opt.input.image_channels, h=self.opt.input.image_size, w=self.opt.input.image_size)
        
        x = self.to_patch_embedding(x)

        z = torch.concatenate([x, y], -1)

        z += self.pos_embedding
        z = self.dropout(z)

        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            
            z = layer(z)

            z = rearrange(z, 'b p d -> b (p d)')
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss
                if torch.isnan(peer_loss):
                    print("WARNING")

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            if torch.isnan(ff_loss):
                print("WARNING")
            z = z.detach()

            if torch.sum(torch.isnan(z)):
                print("WARNING")

            if torch.sum(torch.isnan(z)):
                print("WARNING")
            z = rearrange(z, 'b (p d) -> b p d', p=self.num_patches)

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            x = z[:,:-self.opt.input.num_classes]
            y = z[:,-self.opt.input.num_classes:]
            y = torch.unsqueeze(y, 1)
            y = repeat(y, 'b () d -> b p d', p=self.num_patches)
            x = rearrange(x, 'b (c h w) -> b c h w', c=self.opt.input.image_channels, h=self.opt.input.image_size, w=self.opt.input.image_size)
            x = self.to_patch_embedding(x)
            z = torch.concatenate([x, y], -1)

            z += self.pos_embedding
            z = self.dropout(z)

            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = rearrange(z, 'b p d -> b (p d)')
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)
                    
                z = rearrange(z, 'b (p d) -> b p d', p=self.num_patches)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
