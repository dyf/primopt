import argparse, os
from skimage.draw import ellipse, polygon
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.misc

class Primitive(object):
    def __init__(self, params):
        self.params = np.array(params)

    def __str__(self):
        return str(self.params)

    def mutate(self, d):
        r = 1 + np.random.randn(len(self.params)) * d
        return self.__class__(r * self.params)


class Ellipse(Primitive):
    def render(self, shape):
        x, y, r1, r2, rot, r, g, b, a = self.params
        rr,cc = ellipse(x, y, r1, r2, shape[:2], rot)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [r,g,b,a]
        return im

    def scale(self, f):
        x, y, r1, r2, rot, r, g, b, a = self.params
        return Ellipse([x*f,y*f,r1*f,r2*f, rot, r,g,b,a])
    
    @staticmethod
    def random(shape):
        w = min(shape[:2])
        r = np.random.rand(9)
        return Ellipse(r * np.array([ shape[0], shape[1], w*0.5, w*0.5, 2.0*np.pi, 1.0, 1.0, 1.0, 1.0 ]))

    
class Rectangle(Primitive):
    def render(self, shape):
        x,y,w,h,r,g,b,a = self.params
        p = np.array([ [ x, y ], [ x, y+h ], [ x+w, y+h ], [ x+w, y ] ]).T

        rr,cc = polygon(p[0,:], p[1,:], shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [r,g,b,a]
        return im

    def scale(self, f):
        x,y,w,h,r,g,b,a = self.params
        return Rectangle([x*f, y*f, w*f, h*f, r, g, b, a])

    @staticmethod
    def random(shape):
        r = np.random.rand(8)
        return Rectangle(r * np.array([shape[0], shape[1], shape[0], shape[1], 1.0, 1.0, 1.0, 1.0]))

class RotatedRectangle(Primitive):
    def render(self, shape):
        x,y,w,h,th,r,g,b,a = self.params
        p = np.array([ [ x, y ], [ x, y+h ], [ x+w, y+h ], [ x+w, y ] ]).T
        if th:
            m = np.array([[ np.cos(th), -np.sin(th)], [ np.sin(th), np.cos(th) ]])
            p = np.dot(m, p)
        
        rr,cc = polygon(p[0,:], p[1,:], shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [r,g,b,a]
        return im

    def scale(self, f):
        x,y,w,h,th, r,g,b,a = self.params
        return RotatedRectangle([x*F, y*f, w*f, h*f, th, r, g, b, a])

    @staticmethod
    def random(shape):
        r = np.random.rand(9)
        return RotatedRectangle(r * np.array([shape[0], shape[1], shape[0], shape[1], 2.0*np.pi, 1.0, 1.0, 1.0, 1.0]))


class PrimitiveFactory(object):
    PRIMITIVES = { 'ellipse': Ellipse, 'rotated_rectangle': RotatedRectangle, 'rectangle': Rectangle }

    @staticmethod
    def random(ptype, imshape):
        return PrimitiveFactory.PRIMITIVES[ptype].random(imshape)

    @staticmethod
    def new(ptype, params):
        return PrimitiveFactory.PRIMITIVES[ptype](params)

def optimize_image_levels(target, r_its, m_its, n_prims, levels):
    current = mode_image(target)

    for level in range(levels,-1,-1):
        f = 2 ** level
        target_level = target[::f, ::f, :]
        current_level = current[::f, ::f, :]
        
        for cim, shape, i in optimize_image(target_level, r_its, m_its, n_prims, current_level):
            yield level+1, cim, shape, i
            current = blend_image(current, shape.scale(float(f)).render(target.shape))

        yield 0, current, shape, levels-level

def mode_image(im):
    out = np.ones_like(im)
    out[:,:,0] = scipy.stats.mode(im[:,:,0], axis=None).mode[0]
    out[:,:,1] = scipy.stats.mode(im[:,:,1], axis=None).mode[0]
    out[:,:,2] = scipy.stats.mode(im[:,:,2], axis=None).mode[0]
    return out

def optimize_image(target, r_its, m_its, n_prims, current=None):
    if current is None:
        current = mode_image(target)
    
    for pi in range(n_prims):
        shapes = [ PrimitiveFactory.random('ellipse', target.shape) for i in range(r_its) ]
        errors = [ error_function(s, current, target) for s in shapes ]
        
        best_i = np.argmin(errors)
        best_error = errors[best_i]
        best_shape = shapes[best_i]
                
        next_shape = best_shape
        for mi in range(m_its):
            next_shape = best_shape.mutate(.2)
            error = error_function(next_shape, current, target)
            if error < best_error:
                best_shape = next_shape
                best_error = error            
        
        current = blend_image(current, best_shape.render(target.shape))
        yield current, best_shape, pi
    
def blend_image(current, im):
    alpha_im = im[:,:,3]
    return current * (1 - alpha_im)[:,:,np.newaxis] + im[:,:,:3] * alpha_im[:,:,np.newaxis]

def error_function(s, current, target):    
    im = s.render(target.shape)    
    blend = blend_image(current, im)
    error = np.sqrt(((blend - target) ** 2)).mean()
    return error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('N', type=int)
    parser.add_argument('--r-its', type=int, default=100)
    parser.add_argument('--m-its', type=int, default=100)
    parser.add_argument('--out-dir', default='./out')
    parser.add_argument('--zoom', type=int, default=None)
    parser.add_argument('--levels', type=int, default=1)
    parser.add_argument('--save-its', type=int, default=10)
    args = parser.parse_args()
    
    im = scipy.misc.imread(args.image).astype(float) / 255.0
    if args.zoom:
        im = im[::args.zoom,::args.zoom,:]
    
    if args.levels > 1:
        for level, cim, shape, i in optimize_image_levels(im, args.r_its, args.m_its, args.N, args.levels):
            if level == 0 or i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%02d_%04d.png" % (level, i))
                print(path, str(shape))
                scipy.misc.imsave(path, cim)
    else:
        for cim, shape, i in optimize_image(im, args.r_its, args.m_its, args.N):
            if i % args.save_its == 0:
                path = os.path.join(args.out_dir, "%04d.png" % i)
                print(path, str(shape))
                scipy.misc.imsave(path, cim)

if __name__ == "__main__": main()