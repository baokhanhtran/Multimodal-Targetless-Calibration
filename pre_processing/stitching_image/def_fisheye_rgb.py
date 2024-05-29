#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-1

import os
import argparse

import argcomplete
from tqdm import tqdm

import cv2
from numpy import arange, sqrt, arctan, sin, tan, meshgrid, pi
from numpy import ndarray, hypot

class Defisheye:
    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "angle": 0,
                   "dtype": "equalarea",
                   "format": "fullframe"
                   }
        self._start_att(vkwargs, kwargs)

        if type(infile) == str:
            _image = cv2.imread(infile)
        elif type(infile) == ndarray:
            _image = infile
        else:
            raise Exception("Image format not recognized")

        width = _image.shape[1]
        height = _image.shape[0]
        xcenter = width // 2
        ycenter = height // 2

        dim = min(width, height)
        x0 = xcenter - dim // 2
        xf = xcenter + dim // 2
        y0 = ycenter - dim // 2
        yf = ycenter + dim // 2

        self._image = _image[y0:yf, x0:xf, :]

        self._width = self._image.shape[1]
        self._height = self._image.shape[0]

        if self._xcenter is None:
            self._xcenter = (self._width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (self._height - 1) // 2

    def _map(self, i, j, ofocinv, dim):

        xd = i - self._xcenter
        yd = j - self._ycenter

        rd = hypot(xd, yd)
        phiang = arctan(ofocinv * rd)

        if self._dtype == "linear":
            ifoc = dim * 180 / (self._fov * pi)
            rr = ifoc * phiang
            # rr = "rr={}*phiang;".format(ifoc)

        elif self._dtype == "equalarea":
            ifoc = dim / (2.0 * sin(self._fov * pi / 720))
            rr = ifoc * sin(phiang / 2)
            # rr = "rr={}*sin(phiang/2);".format(ifoc)

        elif self._dtype == "orthographic":
            ifoc = dim / (2.0 * sin(self._fov * pi / 360))
            rr = ifoc * sin(phiang)
            # rr="rr={}*sin(phiang);".format(ifoc)

        elif self._dtype == "stereographic":
            ifoc = dim / (2.0 * tan(self._fov * pi / 720))
            rr = ifoc * tan(phiang / 2)

        rdmask = rd != 0
        xs = xd.copy()
        ys = yd.copy()

        xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + self._xcenter
        ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        return xs, ys

    def convert(self, outfile=None):
        if self._format == "circular":
            dim = min(self._width, self._height)
        elif self._format == "fullframe":
            dim = sqrt(self._width ** 2.0 + self._height ** 2.0)

        if self._radius is not None:
            dim = 2 * self._radius

        ofoc = dim / (2 * tan(self._pfov * pi / 360))
        ofocinv = 1.0 / ofoc

        i = arange(self._width)
        j = arange(self._height)
        i, j = meshgrid(i, j)

        xs, ys, = self._map(i, j, ofocinv, dim)
        img = self._image.copy()

        img[i, j, :] = self._image[xs, ys, :]
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    def _start_att(self, vkwargs, kwargs):
        """
        Starting atributes
        """
        pin = []

        for key, value in kwargs.items():
            if key not in vkwargs:
                raise NameError("Invalid key {}".format(key))
            else:
                pin.append(key)
                setattr(self, "_{}".format(key), value)

        pin = set(pin)
        rkeys = set(vkwargs.keys()) - pin
        for key in rkeys:
            setattr(self, "_{}".format(key), vkwargs[key])

def get_images(input_dir, out_dir, types_images: list = ["png", "jpg", "jpeg"]):
    files = os.listdir(input_dir)
    images = [image for image in files if image.split(".")[-1] in types_images]
    input_images = [os.path.join(input_dir, image) for image in images]
    output_images = [os.path.join(out_dir, image) for image in images]
    return zip(input_images, output_images)

def process_image(input_image, output_image, **kwargs):
    obj = Defisheye(input_image, **kwargs)
    return obj.convert(outfile=output_image)

def batch_process(input_dir, output_dir, **kwargs):
    to_process = get_images(input_dir, output_dir)

    def individual(image_info):
        input_image = image_info[0]
        output_image = image_info[1]
        return process_image(input_image, output_image, **kwargs)

    for in_out_image in tqdm(list(to_process)):
        individual(in_out_image)

def main():
    parser = argparse.ArgumentParser(
        description="Defisheye algorithm")

    parser.add_argument("--image", type=str, default=None,
                        help="Input image to process")

    parser.add_argument("--images_folder", type=str, default=None,
                        help="Input image folder for batch process")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="output directory", required=False)

    parser.add_argument("--fov", type=int, default=180,
                        help="output directory", required=False)

    parser.add_argument("--pfov", type=int, default=120,
                        help="output directory", required=False)

    parser.add_argument("--xcenter", type=int, default=None,
                        help="output directory", required=False)

    parser.add_argument("--ycenter", type=int, default=None,
                        help="output directory", required=False)

    parser.add_argument("--radius", type=int, default=None,
                        help="output directory", required=False)

    parser.add_argument("--angle", type=int, default=0,
                        help="output directory", required=False)

    parser.add_argument("--dtype", type=str, default="equalarea",
                        help="output directory", required=False)

    parser.add_argument("--format", type=str, default="fullframe",
                        help="output directory", required=False)

    argcomplete.autocomplete(parser)

    cfg = parser.parse_args()

    vkwargs = {"fov": cfg.fov,
               "pfov": cfg.pfov,
               "xcenter": cfg.xcenter,
               "ycenter": cfg.ycenter,
               "radius": cfg.radius,
               "angle": cfg.angle,
               "dtype": cfg.dtype,
               "format": cfg.format
               }

    if cfg.image is not None:

        if cfg.save_dir is None:
            normpath = os.path.normpath(cfg.image)
            basedirs = os.path.split(normpath)[0]
            outdir = os.path.join(basedirs[0], "Defisheye")
        else:
            outdir = cfg.save_dir

        os.makedirs(outdir, exist_ok=True)

    elif cfg.images_folder is not None:
        if cfg.save_dir is None:
            normpath = os.path.normpath(cfg.images_folder)
            basedirs = os.path.split(normpath)
            outdir = os.path.join(basedirs[0], basedirs[1] + "-Defisheye")
        else:
            outdir = cfg.save_dir

        os.makedirs(outdir, exist_ok=True)

        batch_process(cfg.images_folder, outdir, **vkwargs)

    else:
        raise Exception(msg="Nor image neither images folder passed.")

    return 0

if __name__ == "__main__":
    main()