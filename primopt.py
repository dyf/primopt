import argparse, os
from skimage.draw import ellipse, polygon, circle
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.misc

class Primitive(object):
    def __init__(self, params, rgb, alpha):
        self.params = np.array(params)
        self.rgb = np.array(rgb)
        self.alpha = float(alpha)
        
    def __str__(self):
        return str(self.params)

    def mutate(self, d):
        r = 1 + np.random.randn(len(self.params)+4) * d
        return self.__class__(r[:-4] * self.params, self.rgb * r[-4:-1], self.alpha * r[-1])

    @staticmethod
    def random_color(target):
        x = np.random.choice(target.shape[0])
        y = np.random.choice(target.shape[1])
        return target[x,y]


class Ellipse(Primitive):
    def render(self, shape):
        x, y, r1, r2, rot = self.params
        rr,cc = ellipse(x, y, r1, r2, shape[:2], rot)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [ self.rgb[0], self.rgb[1],  self.rgb[2], self.alpha ]
        return im

    def scale(self, f):
        x, y, r1, r2, rot = self.params
        return Ellipse([x*f, y*f, r1*f, r2*f, rot], self.rgb, self.alpha)

    @staticmethod
    def random(target):
        w = min(target.shape[:2])
        r = np.random.rand(5)
        return Ellipse(r * np.array([ target.shape[0], target.shape[1], (w+1)*0.5, (w+1)*0.5, 2.0*np.pi]), 
                       Primitive.random_color(target), 
                       np.random.rand())

class Rectangle(Primitive):
    def render(self, shape):
        x,y,w,h = self.params
        p = np.array([ [x, x, x+w, x+w], [y, y+h, y+h, y] ])

        rr,cc = polygon(p[0,:], p[1,:], shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [ self.rgb[0], self.rgb[1],  self.rgb[2], self.alpha ]
        return im

    def scale(self, f):
        x,y,w,h = self.params
        return Rectangle([x*f, y*f, w*f, h*f], self.rgb, self.alpha)

    @staticmethod
    def random(target):
        r = np.random.rand(4)
        return Rectangle(r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1]),
                         Primitive.random_color(target),
                         np.random.rand())

class RotatedRectangle(Primitive):
    def render(self, shape):
        x,y,w,h,th = self.params
        p = np.array([ [x, x, x+w, x+w], [y, y+h, y+h, y] ])
        if th != 0:
            m = np.array([[ np.cos(th), -np.sin(th)], [ np.sin(th), np.cos(th) ]])
            p = np.dot(m, p)
        
        rr,cc = polygon(p[0,:], p[1,:], shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [ self.rgb[0], self.rgb[1],  self.rgb[2], self.alpha ]
        return im

    def scale(self, f):
        x,y,w,h,th = self.params
        return RotatedRectangle([x*f, y*f, w*f, h*f, th], self.rgb, self.alpha)

    @staticmethod
    def random(target):
        r = np.random.rand(5)
        return RotatedRectangle(r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1, 2.0*np.pi]),
                                Primitive.random_color(target), 
                                np.random.rand())
class Circle(Primitive):
    def render(self, shape):
        x,y,r = self.params
        rr,cc = circle(x, y, r, shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [ self.rgb[0], self.rgb[1],  self.rgb[2], self.alpha ]
        return im

    def scale(self, f):
        x,y,r = self.params
        return Circle([x*f,y*f,r*f], self.rgb, self.alpha)

    @staticmethod
    def random(target):
        r = np.random.rand(3)
        return Circle(r * np.array([target.shape[0], target.shape[1], min(target.shape[:2])]),
                      Primitive.random_color(target), 
                      np.random.rand())

class Polygon(Primitive):
    def render(self, shape):
        x,y,r,th = self.params
        angles = np.linspace(0, 2*np.pi, num=self.sides, endpoint=False)
        p =  np.array([ x + r * np.cos(angles), y + r * np.sin(angles) ])

        if th != 0:
            m = np.array([[ np.cos(th), -np.sin(th)], [ np.sin(th), np.cos(th) ]])
            p = np.dot(m, p)

        rr,cc = polygon(p[0,:], p[1,:], shape=shape)
        im = np.zeros( (shape[0], shape[1], 4) )
        im[rr,cc,:] = [ self.rgb[0], self.rgb[1],  self.rgb[2], self.alpha ]

        return im

    def scale(self, f):
        x,y,r,th = self.params
        return self.__class__([x*f,y*f,r*f,th], self.rgb, self.alpha)

    @classmethod
    def random(cls, target):
        r = np.random.rand(4)
        w = np.min(target.shape[:2])
        return cls(r * np.array([ target.shape[0], target.shape[1], w*0.5, 2*np.pi ]),
                   Primitive.random_color(target), 
                   np.random.rand())

class Triangle(Polygon):
    sides = 3
    
class Hexagon(Polygon):
    sides = 6

class PrimitiveFactory(object):
    ELLIPSE = 'ellipse'
    ROTATED_RECTANGLE = 'rotated_rectangle'
    RECTANGLE = 'rectangle'
    CIRCLE = 'circle'
    TRIANGLE = 'triangle'
    HEXAGON = 'hexagon'

    PRIMITIVES = { ELLIPSE: Ellipse, ROTATED_RECTANGLE: RotatedRectangle, RECTANGLE: Rectangle, CIRCLE: Circle, TRIANGLE: Triangle, HEXAGON: Hexagon }

    @staticmethod
    def random(ptype, target):
        return PrimitiveFactory.PRIMITIVES[ptype].random(target)

    @staticmethod
    def new(ptype, params):
        return PrimitiveFactory.PRIMITIVES[ptype](params)

def optimize_image_levels(target, r_its, m_its, n_prims, levels, primitive=PrimitiveFactory.ELLIPSE):
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
                    print "stopping with this level, too many failed images"
                    break
                
                continue
            
            current_error = error
            current = next

            yield current, shape, pi
            pi += 1
            

        print "finished level"

def mode_image(im):
    out = np.ones_like(im)
    out[:,:,0] = scipy.stats.mode(im[:,:,0], axis=None).mode[0]
    out[:,:,1] = scipy.stats.mode(im[:,:,1], axis=None).mode[0]
    out[:,:,2] = scipy.stats.mode(im[:,:,2], axis=None).mode[0]
    return out

def optimize_image(target, r_its, m_its, n_prims, current=None, primitive=PrimitiveFactory.ELLIPSE):
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
    
def blend_image(current, im):
    alpha_im = im[:,:,3]
    return current * (1 - alpha_im)[:,:,np.newaxis] + im[:,:,:3] * alpha_im[:,:,np.newaxis]

def image_error(image, target):    
    return ((image - target) ** 2).mean()
    #return np.sqrt(((image - target) ** 2)).mean()

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
    parser.add_argument('--prim', help="what type of primitive to use", default=PrimitiveFactory.ELLIPSE)
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