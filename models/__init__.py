from .adjusted_unet import AdjustedUNet
from .representation_transformer import RepresentationTransformer
from .ddim import DDIMSamplerGradGuided
from .baselines import *
from .latent_diffusion import *
from .standard_diffusion import *
from .wide_resnet_unet import *
from .wide_resnet import *
from ldm.models.diffusion.ddpm import DDPM


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
    elif name == "joint_diffusion_attention_2_optims":
        return JointDiffusionAttentionDoubleOptims
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
    elif name == "diffmatch_pooling_2_optims":
        return DiffMatchFixedPoolingDoubleOptims
    elif name == "joint_latent_diffusion_noisy_classifier":
        return JointLatentDiffusionNoisyClassifier
    elif name == "joint_latent_diffusion_noisy_attention":
        return JointLatentDiffusionNoisyAttention
    elif name == "joint_latent_diffusion":
        return JointLatentDiffusion
    elif name == "joint_latent_diffusion_attention":
        return JointLatentDiffusionAttention
    elif name == "latent_diffmatch_pooling":
        return LatentDiffMatchPooling
    elif name == "latent_diffmatch_attention":
        return LatentDiffMatchAttention
    elif name == "class_conditioned_ddpm":
        return ClassConditionedDDPM
    elif name == "ddpm":
        return DDPM
    elif name == "resnet18":
        return ResNet
    elif name == "diffmatch_pooling_multilabel":
        return DiffMatchPoolingMultilabel
    elif name == "joint_latent_diffusion_multilabel":
        return JointLatentDiffusionMultilabel
    elif name == "latent_diffmatch_multilabel":
        return LatentDiffMatchPoolingMultilabel
    elif name == "joint_diffusion_knowledge_distillation":
        return JointDiffusionKnowledgeDistillation
    elif name == "ddpm_with_kd":
        return DDPMWithKD
    elif name == "resnet18_with_kd":
        return ResNetWithKD
    else:
        raise NotImplementedError(f"Model {name} is not available")
