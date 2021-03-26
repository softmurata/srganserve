import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from PIL import Image
from model import Generator
from utils import process_image

gpu = 0
upscale_factor = 4

model_path = "SRGAN_DIV2K.pth"

model = Generator()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
torch.cuda.set_device(gpu)
model = model.cuda(gpu)

model.eval()

cudnn.benchmark = True

filename = "cat_lr.jpg"

lr = Image.open(filename)
bicubic = transforms.Resize((lr.size[1] * upscale_factor, lr.size[0] * upscale_factor), Image.BICUBIC)(lr)
lr = process_image(lr, gpu)
bicubic = process_image(bicubic, gpu)


with torch.no_grad():
    sr = model(lr)

images = torch.cat([bicubic, sr], dim=-1)

save_image(images, "cat_hr.png")

