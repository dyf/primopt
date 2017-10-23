import argparse, os
from scipy.ndimage.interpolation import zoom
import numpy as np
import scipy.stats
import scipy.misc
import matplotlib.pyplot as plt
import primitive as primitive

def optimize_image_levels(target, r_its, m_its, n_prims, levels, prim_type=primitive.ELLIPSE):
    current = mode_image(target)

    pi = 1
    for level in range(levels,0,-1):
        f = 2 ** level
        target_level = target[::f,::f,:].copy()
        current_level = current[::f,::f,:].copy()
        #f = float(target.shape[0]) / float(target_level.shape[0]) 
        
        failed_images = 0

        current_error = primitive.image_error(current, target)

        for cim, prim, i in optimize_image(target_level, r_its, m_its, n_prims, current_level, prim_type=prim_type):
            scale_prim = prim.scale(float(f))
            scale_prim.color = prim.color

            next_im = scale_prim.draw(current, target)
            error = primitive.image_error(next_im, target)

            if error > current_error:
                failed_images += 1
                if failed_images > 20:
                    print("stopping with level, too many failed images")
                    break

                continue
            
            current_error = error
            np.copyto(current,next_im)

            yield current, scale_prim, pi
            pi += 1

        print("finished level %d" % level)

    for cim, prim, i in optimize_image(target, r_its, m_its, n_prims, current, prim_type=prim_type):
        yield cim, prim, pi
        pi += 1

def optimize_image(target, r_its, m_its, n_prims, current=None, prim_type=primitive.ELLIPSE):
    if current is None:
        current = mode_image(target)
    else:
        current = current.copy()

    for pi in range(n_prims):       
        current_error = primitive.image_error(current, target)

        best_error = float("inf")
        best_prim = None

        for i in range(r_its):
            prim = primitive.PrimitiveFactory.random(prim_type, target)
            error = prim.error(current, target)
            if error < best_error:
                best_prim = prim
                best_error = error
                
        for i in range(m_its):
            next_prim = best_prim.mutate(.1)
            error = next_prim.error(current, target)
            if error < best_error:
                best_prim = next_prim
                best_error = error
        
        if best_error > current_error:
            continue 

        current = best_prim.draw(current, target)
        yield current, best_prim, pi+1

def mode_image(im):
    out = np.ones_like(im)
    out[:,:,0] = scipy.stats.mode(im[:,:,0], axis=None).mode[0]
    out[:,:,1] = scipy.stats.mode(im[:,:,1], axis=None).mode[0]
    out[:,:,2] = scipy.stats.mode(im[:,:,2], axis=None).mode[0]
    return out

def main():
    parser = argparse.ArgumentParser(description="compose an image from randomized primitives")
    parser.add_argument('image', help="target image to approximate")
    parser.add_argument('N', type=int, help="number of primitives to generate per level of detail")
    parser.add_argument('--r-its', help="number of random iterations to choose next seed primitive", type=int, default=100)
    parser.add_argument('--m-its', help="number of mutation/hill climbing iterations", type=int, default=100)
    parser.add_argument('--out-dir', help="where to save outputs", default='./out')
    parser.add_argument('--zoom', help="zoom level of target image (e.g. optimize a 2x smaller version of input)", type=int, default=None)
    parser.add_argument('--levels', help="number of levels of detail", type=int, default=1)
    parser.add_argument('--save-its', help="how of to save intermediate images (e.g. every 100 frames)", type=int, default=10)
    parser.add_argument('--prim', help="what type of primitive to use", default=primitive.ELLIPSE)
    args = parser.parse_args()
    
    im = scipy.misc.imread(args.image).astype(float) / 255.0
    if args.zoom:
        im = im[::args.zoom,::args.zoom,:]

    if args.levels > 1:
        for cim, prim, i in optimize_image_levels(im, args.r_its, args.m_its, args.N, args.levels, prim_type=args.prim):
            if i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%04d.png" % i)
                print(path, str(prim))
                scipy.misc.imsave(path, cim)
    else:
        for cim, prim, i in optimize_image(im, args.r_its, args.m_its, args.N, prim_type=args.prim):

            if i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%04d.png" % i)
                print(path, str(prim))
                scipy.misc.imsave(path, cim)

if __name__ == "__main__": main()