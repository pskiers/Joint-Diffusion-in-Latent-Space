from .adjusted_unet import AdjustedUNet
from .representation_transformer import RepresentationTransformer
from .ddim import DDIMSamplerGradGuided
from .baselines import *
from .latent_diffusion import *
from .standard_diffusion import *
from .wide_resnet_unet import *
from .wide_resnet import *


def get_model_class(name: str):
    if name == "attention_on_latent_diffusion":
        return AttentionOnLatentDiffusion
    elif name == "wide_resnet":
        return Wide_ResNet
    elif name == "classifier_on_latent_diffusion":
        return ClassifierOnLatentDiffusion
    elif name == "fixmatch":
        return FixMatch
    elif name == "meanmatch":
        return MeanMatch
    elif name == "ddpm_wide_resnet":
        return DDPM_Wide_ResNet
    elif name == "joint_diffusion_noisy_classifier":
        return JointDiffusionNoisyClassifier
    elif name == "joint_diffusion":
        return JointDiffusion
    elif name == "joint_diffusion_attention":
        return JointDiffusionAttention
    elif name == "joint_diffusion_augmentations":
        return JointDiffusionAugmentations
    elif name == "ssl_joint_diffusion":
        return SSLJointDiffusion
    elif name == "ssl_joint_diffusion_attention":
        return SSLJointDiffusionAttention
    elif name == "diffmatch_wideresnet":
        return DiffMatchFixed
    elif name == "diffmatch_attention":
        return DiffMatchFixedAttention
    elif name == "diffmatch_pooling":
        return DiffMatchFixedPooling
    else:
        raise NotImplementedError(f"Model {name} is not available")
