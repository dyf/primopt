import os, argparse
import scipy.misc

import optimize as opt
import primitive
import numpy as np
import primsvg

def save(i, out_dir, cim, init_image, prims):
    im_path = os.path.join(out_dir, "%05d.png" % i)
    svg_path = os.path.join(out_dir, "%05d.svg" % i)
    
    scipy.misc.imsave(im_path, np.clip(cim,0,1))
    primsvg.save(init_image, prims, svg_path)


        
def main():
    parser = argparse.ArgumentParser(description="compose an image from randomized primitives")
    parser.add_argument('image', help="target image to approximate")
    parser.add_argument('N', type=int, help="number of primitives to generate per level of detail")
    parser.add_argument('--r-its', help="number of random iterations to choose next seed primitive", type=int, default=500)
    parser.add_argument('--m-its', help="number of mutation/hill climbing iterations", type=int, default=100)
    parser.add_argument('--out-dir', help="where to save outputs", default='./out')
    parser.add_argument('--zoom', help="zoom level of target image (e.g. optimize a 2x smaller version of input)", type=int, default=None)
    parser.add_argument('--levels', help="number of levels of detail", type=int, nargs='+', default=[0])
    parser.add_argument('--save-its', help="how of to save intermediate images (e.g. every 100 frames)", type=int, default=10)
    parser.add_argument('--prim', help="what type of primitive to use", default='ellipse')
    parser.add_argument('--procs', help="how many processes to use", default=None, type=int)
    parser.add_argument('--init-image', default=None)

    args = parser.parse_args()

    optimizer = opt.PrimitiveOptimizer(r_its=args.r_its,
                                       m_its=args.m_its,
                                       n_prims=args.N,
                                       prim_type=args.prim,
                                       levels=args.levels,
                                       n_procs=args.procs)

    im = scipy.misc.imread(args.image).astype(float) / 255.0
    
    init_image = None
    init_i = 0
    if args.init_image:
        init_image = scipy.misc.imread(args.init_image)
        assert np.allclose(init_image.shape, im.shape)
        base, ext = os.path.splitext(os.path.basename(args.init_image))
        try:
            init_i = int(base)
        except:
            pass    
    else:
        init_image = opt.mean_image(im)

    prims = []
    for i, (im, prim) in enumerate(optimizer.optimize(im, init_image)):
        prims.append(prim)
        if i % args.save_its == 0:
            save(i, args.out_dir, im, init_image, prims)
            

if __name__ == "__main__": main()
