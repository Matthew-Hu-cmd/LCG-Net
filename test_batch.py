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

def load_image(image_path, transform, mode='RGB'):
    """Helper function to load an image and convert it to the specified mode (RGB or L)."""
    image = Image.open(image_path)
    if mode == 'RGB':
        if image.mode != 'RGB':
            image = image.convert('RGB')
    elif mode == 'L':
        if image.mode != 'L':
            image = image.convert('L')
    return transform(image)

# Camouflage function
def camouflage(vgg, decoder, PSF, fore, back, mask):
    b, c, w, h = fore.size()
    down_sam = nn.MaxPool2d((8, 8), (8, 8), (0, 0), ceil_mode=True)
    mask = down_sam(mask)
    fore_f = vgg(fore)
    back_f = vgg(back)
    feat = PSF(fore_f, back_f, mask)
    output = decoder(feat)
    output = output[:, :, :w, :h]
    return output

def embed(fore, mask, back, x, y):
    n_b, c_b, w_b, h_b = back.size()
    n_f, c_f, w_f, h_f = fore.size()

    mask_b = torch.zeros([n_b, 1, w_b, h_b]).to(device)
    fore_b = torch.zeros([n_b, c_b, w_b, h_b]).to(device)

    mask_b[:, :, x:w_f + x, y:h_f + y] = mask
    fore_b[:, :, x:w_f + x, y:h_f + y] = fore
    out = torch.mul(back, 1 - mask_b)
    output = torch.mul(fore_b, mask_b) + out
    return output

# Output the coordinates of the upper left corner of the camouflage region,
# the default camouflage region is in the center of the background image.
def position(fore, back):
    a_s, b_s, c_s, d_s = back.size()
    a_c, b_c, c_c, d_c = fore.size()
    x = abs((c_s - c_c) // 2)
    y = abs((d_s - d_c) // 2)
    return x, y

parser = argparse.ArgumentParser()
# If input by users
parser.add_argument('--fore', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/NC4K/Imgs', help='Foreground images folder.')
parser.add_argument('--mask', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/NC4K/GT', help='Mask images folder.')
parser.add_argument('--back', type=str, default='/home/ac/data/2023/huyang/COD_Dataset/coco_NC4K', help='Background images folder.')

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

# Process all images in the directories
fore_dir = Path(args.fore)
mask_dir = Path(args.mask)
back_dir = Path(args.back)

# Fetch all image files in the directories
fore_images = sorted(list(fore_dir.glob('*.jpg')))
mask_images = sorted(list(mask_dir.glob('*.png')))
back_images = sorted(list(back_dir.glob('*.jpg')))

# Ensure the number of files match across directories
assert len(fore_images) == len(mask_images) == len(back_images), "Mismatch in number of files."

Vertical = 0
Horizontal = 0
Left = 0
Top = 0

decoder = net.decoder
vgg = net.vgg
PSF = net.PSF(in_planes=512)

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

for fore_path, mask_path, back_path in zip(fore_images, mask_images, back_images):
    print(f"Processing {fore_path.name}, {mask_path.name}, {back_path.name}")

    # Ensure foreground and background are RGB, and mask is grayscale
    fore = load_image(fore_path, fore_tf, mode='RGB')
    back = load_image(back_path, back_tf, mode='RGB')
    mask = load_image(mask_path, mask_tf, mode='L')

    Right = fore.size(2)
    Bottom = fore.size(1)

    # Convert images to tensors and move to device
    back = back.to(device).unsqueeze(0)  # Ensure 3 channels
    fore = fore.to(device).unsqueeze(0)  # Ensure 3 channels
    mask = mask.to(device).unsqueeze(0)  # Ensure 1 channel

    # Apply mask thresholding
    mask = (mask > 0).float()
    _, _, w, h = mask.shape

    x, y = position(fore, back)
    Vertical = Vertical if Vertical <= x else x
    Horizontal = Horizontal if Horizontal <= y else y
    x = x + Vertical
    y = y + Horizontal

    back_use = back[:, :, x:x + w, y:y + h]

    with torch.no_grad():
        output_pre = camouflage(vgg, decoder, PSF, fore, back_use, mask)
        output_pre = embed(output_pre, mask, back, x, y)[:, :, Top:Bottom, Left:Right]

    output_name = output_dir / '{:s}_{:s}{:s}'.format(back_path.stem, fore_path.stem, args.save_ext)
    save_image(output_pre, str(output_name))