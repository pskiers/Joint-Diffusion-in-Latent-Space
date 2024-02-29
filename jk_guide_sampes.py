import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm
from omegaconf import OmegaConf
import umap
from models import JointLatentDiffusionMultilabel, MultilabelClassifier
from datasets import ChestXRay_nih_bbox
import torchvision as tv
from ldm.util import default
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def embed_imgs(model, data_loader, max_samples=100000):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list, bbox_list, label_list = [], [], [], []

    for imgs, bbox, label in tqdm(data_loader):
        if len(imgs.shape) == 3:
            imgs = imgs[..., None]
        imgs = rearrange(imgs, 'b h w c -> b c h w')

        with torch.no_grad():
            encoder_posterior = model.encode_first_stage(imgs.to(device))
            latent = model.get_first_stage_encoding(encoder_posterior).detach()
            img_list.append(imgs.cpu())
            embed_list.append(latent.cpu())
            bbox_list.append(bbox)
            label_list.append(label)
        if max_samples is not None and len(img_list) > max_samples:
            break
    return (img_list, embed_list, bbox_list, label_list)


if __name__=='__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from einops import rearrange, repeat
    from ldm.models.diffusion.ddim import DDIMSampler

    # config = OmegaConf.load("logs/a2_jd_classweight0_001_JointLatentDiffusionMultilabel_2024-02-08T16-53-13/configs/config.yaml")
    # config.model.params["ckpt_path"] = f"logs/a2_jd_classweight0_001_JointLatentDiffusionMultilabel_2024-02-08T16-53-13/checkpoints/last.ckpt"

    # config = OmegaConf.load("logs/a7_jd_lr10_4_bcew_JointLatentDiffusionMultilabel_2024-02-09T03-14-54/configs/config.yaml")
    # config.model.params["ckpt_path"] = f"logs/a7_jd_lr10_4_bcew_JointLatentDiffusionMultilabel_2024-02-09T03-14-54/checkpoints/last.ckpt"

    config = OmegaConf.load("logs/a8_jd_lr10_4_14cls_JointLatentDiffusionMultilabel_2024-02-09T07-02-07/configs/config.yaml")
    config.model.params["ckpt_path"] = f"logs/a8_jd_lr10_4_14cls_JointLatentDiffusionMultilabel_2024-02-09T07-02-07/checkpoints/last.ckpt"

    model = JointLatentDiffusionMultilabel(**config.model.get("params", dict()))
    model.sampling_method='conditional_to_x'
    model.to("cuda")
    model.eval()

    model.sample_grad_scale=40

    torch.set_printoptions(sci_mode=False)
    config = OmegaConf.load("/home/jk/Joint-Diffusion-in-Latent-Space/logs/d14_resnet_aug8_100_MultilabelClassifier_2024-02-08T23-03-18/configs/config.yaml")
    ckpt_path = f"/home/jk/Joint-Diffusion-in-Latent-Space/logs/d14_resnet_aug8_100_MultilabelClassifier_2024-02-08T23-03-18/checkpoints/last.ckpt"

    model_2 = MultilabelClassifier(**config.model.get("params", dict()))
    checkpoint = torch.load(ckpt_path)
    model_2.load_state_dict(checkpoint["state_dict"])
    model_2.to("cuda")
    model_2.eval()

    cl_list = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis", "Hernia","Infiltration", "Mass", "Nodule","Pleural_Thickening","Pneumonia","Pneumothorax","No Finding"]
    cl_list_loop=cl_list[2:]#.copy()
    T = 500
    num_timesteps = 1000

    for class_ in cl_list_loop:
        dataset = ChestXRay_nih_bbox(pick_class=class_)
        if len(dataset)>0:
            print(class_, 'has N samples: ', len(dataset))
            dl = torch.utils.data.DataLoader(dataset, batch_size=min(len(dataset), 48), shuffle=False)

            ret = embed_imgs(model, dl)
            batch = 0
            x_samples_save = []
            img_original_save = []
            pred_o_save = []
            pred_o_ext_save = []
            pred_ext_save = []
            bbox_save = []


            for img_original, z, bbox in zip(ret[0], ret[1], ret[2]):
                # denoise samples
                z = z.to("cuda")
                t = repeat(torch.tensor([T]), '1 -> b', b=len(z))
                t = t.to("cuda").long()
                noise = torch.randn_like(z)
                z_noisy = model.q_sample(x_start=z, t=t, noise=noise)
                shape = z_noisy.shape            
                samples, pred_o = model.p_sample_loop(cond=None, shape = shape, original_img = z, 
                                                    return_intermediates=False, x_T=z_noisy, start_T=T, 
                                                    pick_class=class_, return_pred_o=True)
                x_samples = model.decode_first_stage(samples)
                
                #predictions with external clasisfier
                img_original = img_original.to("cuda")
                pred_o_ext = model_2(img_original)
                pred_ext = model_2(x_samples)

                x_samples_save.append(x_samples.detach().cpu().clone())
                del x_samples
                img_original_save.append(img_original.detach().cpu().clone())
                del img_original
                pred_o_save.append(pred_o.detach().cpu().clone())
                del pred_o
                pred_o_ext_save.append(pred_o_ext.detach().cpu().clone())
                del pred_o_ext
                pred_ext_save.append(pred_ext.detach().cpu().clone())
                del pred_ext
                bbox_save.append(bbox.clone())
                del bbox


                batch+=1
                if batch>1000:
                    break
                
            folder_to_save = "vce_results_baseline"
            torch.save(torch.cat(x_samples_save, dim=0), f'{folder_to_save}/T{T}_{class_}_x_samples.pt')
            del x_samples_save
            torch.save(torch.cat(img_original_save, dim=0), f'{folder_to_save}/T{T}_{class_}_img_original.pt')
            del img_original_save
            torch.save(torch.cat(pred_o_save, dim=0), f'{folder_to_save}/T{T}_{class_}_pred_o.pt')
            del pred_o_save
            torch.save(torch.cat(pred_o_ext_save, dim=0), f'{folder_to_save}/T{T}_{class_}_pred_o_ext.pt')
            del pred_o_ext_save
            torch.save(torch.cat(pred_ext_save, dim=0), f'{folder_to_save}/T{T}_{class_}_pred_ext.pt')
            del pred_ext_save
            torch.save(torch.cat(bbox_save, dim=0), f'{folder_to_save}/T{T}_{class_}_bbox.pt')
            del bbox_save
                


