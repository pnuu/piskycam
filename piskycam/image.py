import numpy as np

from PIL import Image


def yuv2rgb(data):
    """Convert *data* from YUV to RGB."""
    itu = np.array([[1.164,  0.000,  1.596], [1.164, -0.392, -0.813], [1.164,  2.017,  0.000]])
    yuv = np.array(data).astype(np.float32)
    yuv[:, :, 0] -= 16
    yuv[:, :, 1:] -= 128
    return yuv.dot(itu.T).clip(0, 255).astype(np.uint8)


def data2img(data):
    """Convert *data* to image."""
    return Image.fromarray(data, 'RGB')


def save_max(fid, fname):
    """Save max stack."""
    data = yuv2rgb(fid["max"])
    img = data2img(data)
    img.save(fname)


def save_ave(fid, fname):
    """Save average stack."""
    data = yuv2rgb(fid["sum"] / np.array(fid["count"]))
    img = data2img(data)
    img.save(fname)
