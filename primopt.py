import argparse, os
from skimage.transform import resize
import numpy as np
import scipy.stats
import scipy.misc

import primitive as prim

def optimize_image_levels(target, r_its, m_its, n_prims, levels, primitive=prim.ELLIPSE):
    current = mode_image(target)

    pi = 0
    for level in range(levels,-1,-1):
        f = int(2 ** level)
        target_level = target[::f,::f,:]
        current_level = current[::f,::f,:]
        failed_images = 0

        current_error = image_error(current, target)

        for cim, shape, i in optimize_image(target_level, r_its, m_its, n_prims, current_level, primitive=primitive):
            scale_shape = shape.scale(float(f))
            
            next = blend_image(current, scale_shape.render(target.shape))
            error = image_error(next, target)
            
            if error > current_error:
                failed_images += 1
                if failed_images > 20:
                    print("stopping with level, too many failed images")
                    break

                continue
            
            current_error = error
            current = next

            yield current, shape, pi
            pi += 1
            

        print("finished level %d" % level)

def optimize_image(target, r_its, m_its, n_prims, current=None, primitive=prim.ELLIPSE):
    if current is None:
        current = mode_image(target)
    
    for pi in range(n_prims):       

        shapes = [ PrimitiveFactory.random(primitive, target) for i in range(r_its) ]
        errors = [ shape_error(s, current, target) for s in shapes ]
        
        best_i = np.argmin(errors)
        best_error = errors[best_i]
        best_shape = shapes[best_i]
                
        next_shape = best_shape
        for mi in range(m_its):
            next_shape = best_shape.mutate(.2)
            error = shape_error(next_shape, current, target)
            if error < best_error:
                best_shape = next_shape
                best_error = error            

        current_error = image_error(current, target)
        if best_error > current_error:
            continue 
        
        current = blend_image(current, best_shape.render(target.shape))
        yield current, best_shape, pi

def mode_image(im):
    out = np.ones_like(im)
    out[:,:,0] = scipy.stats.mode(im[:,:,0], axis=None).mode[0]
    out[:,:,1] = scipy.stats.mode(im[:,:,1], axis=None).mode[0]
    out[:,:,2] = scipy.stats.mode(im[:,:,2], axis=None).mode[0]
    return out
    
def blend_image(current, im):
    alpha_im = im[:,:,3]
    return current * (1 - alpha_im)[:,:,np.newaxis] + im[:,:,:3] * alpha_im[:,:,np.newaxis]

def image_error(image, target):    
    return ((image - target) ** 2).mean()

def shape_error(s, current, target):    
    im = s.render(target.shape)    
    blend = blend_image(current, im)
    return image_error(blend, target)
    
def main():
    parser = argparse.ArgumentParser(description="compose an image from randomized primitives")
    parser.add_argument('image', help="target image to approximate")
    parser.add_argument('N', type=int, help="number of primitives to generate per level of detail")
    parser.add_argument('--r-its', help="number of random iterations to choose next seed primitive", type=int, default=500)
    parser.add_argument('--m-its', help="number of mutation/hill climbing iterations", type=int, default=100)
    parser.add_argument('--out-dir', help="where to save outputs", default='./out')
    parser.add_argument('--zoom', help="zoom level of target image (e.g. optimize a 2x smaller version of input)", type=int, default=None)
    parser.add_argument('--levels', help="number of levels of detail", type=int, default=1)
    parser.add_argument('--save-its', help="how of to save intermediate images (e.g. every 100 frames)", type=int, default=100)
    parser.add_argument('--prim', help="what type of primitive to use", default=prim.ELLIPSE)
    args = parser.parse_args()
    
    im = scipy.misc.imread(args.image).astype(float) / 255.0
    if args.zoom:
        im = im[::args.zoom,::args.zoom,:]
    
    if args.levels > 1:
        for cim, shape, i in optimize_image_levels(im, args.r_its, args.m_its, args.N, args.levels, primitive=args.prim):
            if i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%04d.png" % i)
                print(path, str(shape))
                scipy.misc.imsave(path, cim)
    else:
        for cim, shape, i in optimize_image(im, args.r_its, args.m_its, args.N, primitive=args.prim):
            if i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%04d.png" % i)
                print(path, str(shape))
                scipy.misc.imsave(path, cim)

if __name__ == "__main__": main()