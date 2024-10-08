import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from params_position import example_all

device = torch.device('cuda:3')

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#Camoufalge function
def camouflage(vgg, decoder, PSF, fore, back, mask):
    b,c,w,h = fore.size()
    down_sam = nn.MaxPool2d((8, 8), (8, 8), (0, 0), ceil_mode=True)
    mask = down_sam(mask)
    fore_f = vgg(fore)
    back_f = vgg(back)
    feat = PSF(fore_f,back_f,mask)
    output = decoder(feat)
    output = output[:,:,:w,:h]
    return output

def embed(fore,mask,back,x,y):
    n_b, c_b, w_b, h_b = back.size()
    n_f, c_f, w_f, h_f = fore.size()

    mask_b = torch.zeros([n_b, 1, w_b, h_b]).to(device)
    fore_b = torch.zeros([n_b, c_b, w_b, h_b]).to(device)

    mask_b[:,:,x:w_f + x, y : h_f+y] = mask
    fore_b[:,:, x:w_f+x, y : h_f+y] = fore
    out = torch.mul(back, 1-mask_b)
    output = torch.mul(fore_b, mask_b) + out
    return output

# Output the coordinates of the upper left corner of the camouflage region,
# the default camouflage region is in the center of the background image.
def position(fore, back):
    a_s, b_s, c_s, d_s = back.size()
    a_c, b_c, c_c, d_c = fore.size()
    x = abs((c_s - c_c) // 2)
    y = abs((d_s - d_c) // 2)
    return x,y

parser = argparse.ArgumentParser()
# If input by users
parser.add_argument('--fore', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/NC4K/Imgs/1002.jpg', help='Foreground image.')
parser.add_argument('--mask', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/NC4K/GT/1002.png', help='Mask image.')
parser.add_argument('--back', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/coco_NC4K/1002.jpg', help='Background image.')

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--PSF', type=str, default='models/PSF.pth')

# Additional options
parser.add_argument('--fore_size', type=int, default=0,
                    help='New (minimum) size for the fore image, \
                    keeping the original size if set to 0')
parser.add_argument('--back_size', type=int, default=0,
                    help='New (minimum) size for the back image, \
                    keeping the original size if set to 0')
parser.add_argument('--mask_size', type=int, default=0,
                    help='New (minimum) size for the mask image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output/NC4K',
                    help='Directory to save the output image(s)')
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)


assert (args.fore)
fore_path = [Path(args.fore)]
assert (args.mask)
mask_path = [Path(args.mask)]
assert (args.back)
back_path = [Path(args.back)]

Vertical = 0
Horizontal =0
Left = 0
Top = 0

decoder = net.decoder
vgg = net.vgg
PSF = net.PSF(in_planes = 512)

decoder.eval()
vgg.eval()
PSF.eval()

decoder.load_state_dict(torch.load(args.decoder, weights_only=True))
PSF.load_state_dict(torch.load(args.PSF, weights_only=True))
vgg.load_state_dict(torch.load(args.vgg, weights_only=True))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)
PSF.to(device)

fore_tf = test_transform(args.fore_size, args.crop)
back_tf = test_transform(args.back_size, args.crop)
mask_tf = test_transform(args.mask_size, args.crop)

for (fore_path,mask_path) in zip(fore_path, mask_path):
    for back_path in back_path:

        fore = Image.open(str(fore_path))
        back = Image.open(str(back_path))

        # If the foreground is larger than the background, scale the foreground to the background size.
        tempSize = [fore.size[0], fore.size[1]]
        if tempSize[0] > back.size[0]:
            tempSize[0] = back.size[0]
            tempSize[1] = int(tempSize[1] * back.size[0] /(fore.size[0]))
        if tempSize[1] > back.size[1]:
            temp = tempSize[1]
            tempSize[1] = back.size[1]
            tempSize[0] = int(tempSize[0] * back.size[1] / (temp))

        fore_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)
        mask_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)

        fore = fore_tf(fore)
        back = back_tf(back)
        mask = mask_tf(Image.open(str(mask_path)))
        Right = tempSize[0]
        Bottom = tempSize[1]

        back = back.to(device).unsqueeze(0)
        fore = fore.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)

        mask = (mask>0).float()
        _,_,w,h =mask.shape

        x, y = position(fore, back)
        Vertical = Vertical if Vertical<=x else x
        Horizontal = Horizontal if Horizontal<=y else y
        x = x + Vertical
        y = y + Horizontal

        back_use = back[:, :, x:x + w, y:y + h]

        with torch.no_grad():
            output_pre = camouflage(vgg, decoder, PSF, fore, back_use, mask)
            output_pre = embed(output_pre, mask, back, x, y)[:,:,Top:Bottom,Left:Right]
        output_name = output_dir / '{:s}_{:s}{:s}'.format(back_path.stem, fore_path.stem, args.save_ext)
        save_image(output_pre, str(output_name))

