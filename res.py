import os
import glob
from PIL import Image
import tqdm
list_ = glob.glob("/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/data/images_*/images/*")

for i in tqdm.tqdm(list_):
    img = Image.open(i)
    img2 = img.resize((256,256))
    img2.save("/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/data/images256/"+os.path.basename(i))