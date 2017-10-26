import os, argparse
import scipy.misc

import optimize as opt
import primitive
import numpy as np

def save(i, save_its, out_dir, cim):
    if i % save_its == 0:
        path = os.path.join(out_dir, "%05d.png" % i)
        print(path)
        scipy.misc.imsave(path, np.clip(cim,0,1))
        
def main():
    parser = argparse.ArgumentParser(description="compose an image from randomized primitives")
    parser.add_argument('image', help="target image to approximate")
    parser.add_argument('N', type=int, help="number of primitives to generate per level of detail")
    parser.add_argument('--r-its', help="number of random iterations to choose next seed primitive", type=int, default=500)
    parser.add_argument('--m-its', help="number of mutation/hill climbing iterations", type=int, default=100)
    parser.add_argument('--out-dir', help="where to save outputs", default='./out')
    parser.add_argument('--zoom', help="zoom level of target image (e.g. optimize a 2x smaller version of input)", type=int, default=None)
    parser.add_argument('--levels', help="number of levels of detail", type=int, default=1)
    parser.add_argument('--save-its', help="how of to save intermediate images (e.g. every 100 frames)", type=int, default=10)
    parser.add_argument('--prim', help="what type of primitive to use", default=primitive.ELLIPSE)
    parser.add_argument('--procs', help="how many processes to use", default=None, type=int)

    args = parser.parse_args()

    opt.init_pool(args.procs)

    im = scipy.misc.imread(args.image).astype(float) / 255.0
    if args.zoom:
        im = im[::args.zoom,::args.zoom,:]

    if args.levels > 1:
        for cim, prim, i in opt.optimize_image_levels(im, args.r_its, args.m_its, args.N, args.levels, prim_type=args.prim):
            save(i, args.save_its, args.out_dir, cim)
    else:
        for cim, prim, i in opt.optimize_image(im, args.r_its, args.m_its, args.N, prim_type=args.prim):
            save(i, args.save_its, args.out_dir, cim)

if __name__ == "__main__": main()
