from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt

from scipy.linalg import eigh as scipy_eigh
import torch
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale


PATH_DSET = kagglehub.dataset_download("maciejgronczynski/biggest-genderface-recognition-dataset")
PATH_IMGS = [
    f"{PATH_DSET}/faces/man/man_12600.jpg",
    f"{PATH_DSET}/faces/man/man_13600.jpg",
    f"{PATH_DSET}/faces/woman/woman_7600.jpg",
]
PATH_TENSOR = f'{Path(__file__).resolve().parent}/tensors'
IMG_SHAPE = (1, 250, 250)
N_DIMS = 62500
K_EVECS = 1500


# ### generate the eigenvectors: ###

def gen():
    dset = ImageFolder(
        PATH_DSET,
        loader=lambda x: load_img(x),
        is_valid_file=lambda x: list(read_image(x).shape) == [3, 250, 250],
        allow_empty=True,
    )
    
    ts = torch.stack([p[0].flatten() for p in dset])  # [13230 x 62500]
    
    t_avg = ts.mean(0, False)
    save("t_avg", t_avg)
    ts -= t_avg
    
    t_std = ts.std(0, False)
    save("t_std", t_std)
    ts /= t_std
    
    t_cov = torch.cov(ts.T)
    save("t_cov", t_cov)
    
    n = t_cov.shape[0]
    (e_vals, e_vecs) = scipy_eigh(t_cov.numpy(), driver="evx", overwrite_a=True, subset_by_index=[n-1024, n-1])
    save("e_vals", torch.tensor(e_vals))
    save("e_vecs", torch.tensor(e_vecs))


# ### applications of the eigenvectors: ###

def deflate(t_avg, t_std, e_vecs, t_img):
    print(t_img.shape, t_avg.shape)
    t2 = (t_img.flatten() - t_avg) / t_std
    t2 = t2.reshape((N_DIMS))
    return e_vecs @ t2


def inflate(t_avg, t_std, e_vecs, t0):
    t3 = t0.unsqueeze(0) @ e_vecs
    t3 = t3 * t_std + t_avg
    t3 = t3.clip(0, 255)
    return torch.reshape(t3, IMG_SHAPE)


def deflate_inflate(t_avg, t_std, e_vecs, t_before):
    t_small = deflate(t_avg, t_std, e_vecs, t_before)
    t_after = inflate(t_avg, t_std, e_vecs, t_small)
    return [t_before, t_after]


def test_compression():
    t_avg = load("t_avg")
    t_std = load("t_std")
    e_vecs = load("e_vecs").T[-K_EVECS:]

    ps = [deflate_inflate(t_avg, t_std, e_vecs, load_img(path)) for path in PATH_IMGS]
    ps = [x for p in zip(*ps) for x in p]
    collage(ps, 3)
        

def show_eigenfaces():
    e_vecs = load("e_vecs").T[-25:].flip(0)
    e_vecs = torch.reshape(e_vecs, (e_vecs.shape[0], *IMG_SHAPE))
    collage([*e_vecs], 5)


def show_statsfaces():
    t_avg = load("t_avg")
    t_std = load("t_std")
    imgs = [t_avg, t_std, (t_avg / t_std)]
    collage(imgs, 1)

# ### utils: ###

def save(name, t):
    print(f'saving tensor {name} - {t.shape}...')
    torch.save(t, f'{PATH_TENSOR}/{name}.pt')


def load(name):
    print(f'loading tensor {name}...')
    return torch.load(f'{PATH_TENSOR}/{name}.pt')


def load_img(path):
    return rgb_to_grayscale(read_image(path).float())


def rescale(img):
    t = torch.reshape(img, IMG_SHAPE)
    t = t - t.min()
    return t / t.max()


def collage(imgs, cols):
    ts = torch.stack([rescale(x) for x in imgs])
    img_grid = make_grid(ts, nrow=cols)
    plt.imshow(img_grid[0], cmap="gray")
    plt.show()


if __name__ == '__main__':
    # gen()  # this can take around 1 hour to run
    test_compression()
    show_eigenfaces()
    show_statsfaces()
    print("done.")

