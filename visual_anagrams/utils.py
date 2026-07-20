import pickle
from pathlib import Path

from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


def add_args(parser):
    '''
    Add arguments for sampling to a parser
    '''

    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--save_dir", type=str, default='results', help='Location to samples and metadata')
    parser.add_argument("--prompts", required=True, type=str, nargs='+', help='Prompts to use, corresponding to each view.')
    parser.add_argument("--views", required=True, type=str, nargs='+', help='Name of views to use. See `get_views` in `views.py`.')
    parser.add_argument("--style", default='', type=str, help='Optional string to prepend prompt with')
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--noise_level", type=int, default=50, help='Noise level for stage 2')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_metadata", action='store_true', help='If true, save metadata about the views. May use lots of disk space, particularly for permutation views.')
    parser.add_argument("--generate_1024", action='store_true', help='If true, also upsample using SD 4x upsampler. Please see readme for more details.')
    parser.add_argument("--view_args", default=None, type=str, nargs='+', help='Args to pass to views')
    parser.add_argument("--ref_im_path", type=str, default=None, help='Image from which to extract a component, when solving inverse problems')

    return parser




def load_illusion(path, size, device, dtype=torch.float16):
    '''
    Load an illusion image from disk for SDEdit / img2img re-noising.

    path (str) :
        Path to an image (e.g. a prior `sample_64.png` / `sample_256.png`)

    size (int) :
        Spatial size to resize to (e.g. 64 for stage 1, 256 for stage 2)

    Returns a tensor of shape (1,3,size,size), in [-1,1], on `device`.
    '''
    im = Image.open(path).convert('RGB')
    im = TF.to_tensor(im) * 2 - 1                       # (3,H,W) in [-1,1]
    im = TF.resize(im, (size, size), antialias=True)    # illusions are square
    return im[None].to(device=device, dtype=dtype)      # (1,3,size,size)

def save_illusion(image, views, sample_dir):
    '''
    Saves the illusion (`image`), as well as all views of the illusion

    image (torch.tensor) :
        Tensor of shape (1,3,H,W) representing the image

    views (views.BaseView) :
        Represents the view, inherits from BaseView

    sample_dir (pathlib.Path) :
        pathlib Path object, representing the directory to save to
    '''

    size = image.shape[-1]

    # Save illusion
    save_name = sample_dir / f'sample_{size}.png'
    save_image(image / 2. + 0.5, save_name, padding=0)

    # Save views of the illusion
    # TODO: This is kind of ugly... can we fix it?
    im_views = torch.stack([view.save_view(image[0]) 
                            if hasattr(view, 'save_view') 
                            else view.view(image[0]) 
                            for view in views])
    save_name_views = sample_dir / f'sample_{size}.views.png'
    save_image(im_views / 2. + 0.5, save_name_views, padding=0)

def save_metadata(views, args, save_dir):
    '''
    Saves the following the sample_dir
        1) pickled view object
        2) args for the illusion
    '''

    metadata = {
        'views': views,
        'args': args
    }
    with open(save_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def get_courier_font_path():
    font_path = Path(__file__).parent / 'assets' / 'CourierPrime-Regular.ttf'
    return str(font_path)