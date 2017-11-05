from skimage.draw import ellipse, polygon, circle, line, bezier_curve
from skimage.filters import gabor_kernel
import numpy as np

ADD = 'add'
COMPOSITE = 'composite'
COLOR_FROM_TARGET = 'color_from_target'

def image_error(image, target):        
    return ((image - target) ** 2).mean()

def rot2d(th):
    return np.array([[ np.cos(th), -np.sin(th)], 
                     [ np.sin(th), np.cos(th) ]])

class Primitive(object):
    def __init__(self, params, alpha=None, color=COLOR_FROM_TARGET, mutate_alpha=True, mutate_color=False):
        self.params = np.array(params)
        self.alpha = float(alpha) if alpha is not None else 1.0
        self.color = color
        self.mode = COMPOSITE
        self.mutate_alpha = mutate_alpha
        self.mutate_color = mutate_color

    def error(self, current, target):
        blend = self.draw(current, target)
        return image_error(blend, target)
        
    def select_color(self, current, target, px):
        if len(px[0]) == 0:
            return self.color
        idx = np.random.choice(len(px[0]))
        return target[px[0][idx], px[1][idx], :].copy()

    def mutate(self, d):
        params = self.params * (1 + np.random.randn(len(self.params)) * d)                
        alpha = self.alpha if not self.mutate_alpha else self.alpha * (1 + np.random.randn(1)) * d
        color = self.color if not self.mutate_color else self.color * (1 + np.random.randn(3)) * d
        return self.__class__(params, alpha=alpha, color=color, mutate_alpha=self.mutate_alpha, mutate_color=self.mutate_color)

    @classmethod
    def random(cls, target, **kwargs):        
        if kwargs.get('alpha', None) is None:
            kwargs['alpha'] = np.random.rand() 
        return cls(cls.random_params(target), **kwargs)



class ShapePrimitive(Primitive):

    def draw(self, current, target):
        
        mask_px = self.rasterize(target.shape)

        out = current.copy()

        if self.color is COLOR_FROM_TARGET:
            self.color = self.select_color(current, target, mask_px)        
            if self.color is COLOR_FROM_TARGET:
                return out

        if self.mode == COMPOSITE:
            out[mask_px[0], mask_px[1], :] = out[mask_px[0], mask_px[1], :] * (1.0 - self.alpha) + self.color * self.alpha
        elif self.mode == ADD:
            out[mask_px[0], mask_px[1], :] += self.color * self.alpha
        return out

class ImagePrimitive(Primitive):
    def draw(self, current, target):
        image = self.rasterize(target.shape)
        mask_px = np.where(image != 0)

        out = current.copy()

        if self.color is COLOR_FROM_TARGET:
            self.color = self.select_color(current, target, mask_px)        
        
            if self.color is COLOR_FROM_TARGET:
                return out

        if self.mode == COMPOSITE:
            out[mask_px[0], mask_px[1], :] = out[mask_px[0], mask_px[1], :] * (1.0 - self.alpha) + self.color * self.alpha
        elif self.mode == ADD:
            out[mask_px[0], mask_px[1], :] += self.color * self.alpha
        return out    

class Crescent(ImagePrimitive):
    def rasterize(self, shape):
        x, y, r1, r2, th, offset = self.params                
        e1 = ellipse(x, y, r1, r2, shape[:2], th)

        x2,y2 = x + offset*np.cos(th), y + offset*np.sin(th)

        e2 = ellipse(x2, y2, r1, r2, shape[:2], th)

        im = np.zeros(shape[:2], dtype=float)
        im[e1] = 1
        im[e2] = 0
        
        return im

    @staticmethod
    def random_params(target):
        r = np.random.rand(6)
        return np.array([target.shape[0]*r[0], target.shape[1]*r[1], 
                         target.shape[0]*.5*r[2], target.shape[1]*.5*r[3], 
                         2*np.pi*r[4], (r[5]-.5)*r[2]*target.shape[0]])

    def scale(self, f):
        x, y, r1, r2, th, offset = self.params        
        return Crescent([x*f, y*f, r1*f, r2*f, th, offset*f], self.alpha)

class Gaussian(ImagePrimitive):
    def rasterize(self, shape):
        x, y, sx, sy, th = self.params
        w = 3 * max(sx, sy) * np.sqrt(2)
        h = w

        xx,yy = np.mgrid[0:w,0:h]
        if th != 0:
            st = np.sin(th)
            ct = np.cos(th)
            xr = xx*ct - yy*st
            yr = xx*st + yy*ct  
            xx, yy = xr, yr
        
        kernel = numpy.exp(-(((w*0.5-xx)/sx)**2+((h*0.5-yy)/sy)**2)/2.)
        
        dx = x + kernel.shape[0] - shape[0]
        dy = y + kernel.shape[1] - shape[1]
        if dx > 0:
            kernel = kernel[:kernel.shape[0]-dx,:]
        if dy > 0:
            kernel = kernel[:,:kernel.shape[1]-dy]
        
        im = np.zeros(shape)
        im[x:x+kernel.shape[0],y:y+kernel.shape[1]] += kernel

        return im

    @staticmethod
    def random_params(target):
        r = np.random.rand(5)
        return r * np.array([target.shape[0], target.shape[1], target.shape[0]*.5, target.shape[1]*.5, 2*np.pi])

    def scale(self, f):
        x, y, sx, sy, th = self.params      
        return Gaussian([x*f, y*f, sx*f, sy*f, th], self.alpha)



class Gabor(ImagePrimitive):
    def __init__(self, *args, **kwargs):
        super(Gabor, self).__init__(*args, **kwargs)
        self.mode = ADD

    def rasterize(self, shape):
        im = np.zeros((shape[0],shape[1]))

        x, y, frequency, theta, sigma = self.params
        x,y = int(x),int(y)
        
        if x >= shape[0] or y >= shape[1]:
            return im

        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=4, sigma_y=4))
    
        dx = x + kernel.shape[0] - shape[0]
        dy = y + kernel.shape[1] - shape[1]
        if dx > 0:
            kernel = kernel[:kernel.shape[0]-dx,:]
        if dy > 0:
            kernel = kernel[:,:kernel.shape[1]-dy]
        
        im[x:x+kernel.shape[0],y:y+kernel.shape[1]] += kernel        

        return im

    @staticmethod
    def random_params(target):
        r = np.random.rand(3)
        return r * np.array([np.random.randint(2,target.shape[0]-2), np.random.randint(2,target.shape[1]-2),
                            r[0] + (1.0 - r[0]) / target.shape[0], r[1]*np.pi, 
                            r[2]*target.shape[0]])

    def mutate(self, d):
        r = 1 + np.random.randn(len(self.params)+4) * d
        return self.__class__(r[:-4] * self.params, self.alpha * r[-4], self.color*r[-3:])


class Cosine(ImagePrimitive):
    def rasterize(self, shape):
        sfx, sfy, ax, ay = self.params
        xx,yy = np.mgrid[0:shape[0],0:shape[1]]
        cx = 2 * ax * np.cos(np.pi*xx*(2*sfx+1)/shape[0])
        cy = 2 * ay * np.cos(np.pi*yy*(2*sfy+1)/shape[1])
        return cx * cy

    def scale(self, f):        
        sfx,xfy, ax, ay = self.params.shape
        return Cosine([sfx*f, sfy*f, ax, ay], self.alpha)

    @staticmethod
    def random_params(target):
        r = np.random.rand(4)
        return r * np.array([target.shape[0], target.shape[1], target.shape[0], target.shape[1]])
                      

class Line(ShapePrimitive):
    def rasterize(self, shape):
        x1, y1, x2, y2 = self.params
        x1, x2 = np.clip([x1,x2], 0, shape[0]-1).astype(int)
        y1, y2 = np.clip([y1,y2], 0, shape[1]-1).astype(int)
        
        return line(x1, y1, x2, y2)

    def scale(self, f):
        x1, y1, x2, y2 = self.params
        return Line([x1*f, y1*f, x2*f, y2*f], self.alpha)

    @staticmethod
    def random_params(target):
        r = np.random.rand(4)
        return r * np.array([ target.shape[0]-1, target.shape[1]-1, target.shape[0]-1, target.shape[1]-1 ])

class Bezier(ShapePrimitive):
    def rasterize(self, shape):
        x1, y1, x2, y2, x3, y3, weight = self.params
        x1, x2, x3 = np.clip([x1,x2,x3], 0, shape[0]-1).astype(int)
        y1, y2, y3 = np.clip([y1,y2,y3], 0, shape[1]-1).astype(int)
        
        return bezier_curve(x1, y1, x2, y2, x3, y3, weight, shape=shape )

    def scale(self, f):        
        return Bezier(self.params * f, self.alpha)

    @staticmethod
    def random_params(target):
        r = np.random.rand(7)
        return r * np.array([ target.shape[0]-1, target.shape[1]-1, 
                              target.shape[0]-1, target.shape[1]-1,
                              target.shape[0]-1, target.shape[1]-1, 8.0 ])
  
class Ellipse(ShapePrimitive):
    def rasterize(self, shape):
        x, y, r1, r2, rot = self.params
        return ellipse(x, y, r1, r2, shape[:2], rot)

    def scale(self, f):
        x, y, r1, r2, rot = self.params
        return Ellipse([x*f, y*f, r1*f, r2*f, rot], self.alpha)

    @staticmethod
    def random_params(target):
        w = min(target.shape[:2])
        r = np.random.rand(5)
        return r * np.array([ target.shape[0], target.shape[1], (w+1)*0.5, (w+1)*0.5, 2.0*np.pi ])

class Rectangle(ShapePrimitive):
    def rasterize(self, shape):
        x,y,w,h = self.params
        p = np.array([ [x, x, x+w, x+w], [y, y+h, y+h, y] ])
        return polygon(p[0,:], p[1,:], shape=shape)

    def scale(self, f):
        x,y,w,h = self.params
        return Rectangle([x*f, y*f, w*f, h*f], self.alpha, self.color)

    @staticmethod
    def random_params(target):
        r = np.random.rand(4)
        return r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1])

class RotatedRectangle(ShapePrimitive):
    def rasterize(self, shape):
        x,y,w,h,th = self.params
        p = np.array([ [x, x, x+w, x+w], [y, y+h, y+h, y] ])
        if th != 0:            
            p = np.dot(rot2d(th), p)
        
        return polygon(p[0,:], p[1,:], shape=shape)
        
    def scale(self, f):
        x,y,w,h,th = self.params
        return RotatedRectangle([x*f, y*f, w*f, h*f, th], self.alpha, self.color)

    @staticmethod
    def random_params(target):
        r = np.random.rand(5)
        return r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1, 2.0*np.pi])

class Circle(ShapePrimitive):
    def rasterize(self, shape):
        x,y,r = self.params
        return circle(x, y, r, shape=shape)

    def scale(self, f):
        x,y,r = self.params
        return Circle([x*f,y*f,r*f], self.alpha, self.color)

    @staticmethod
    def random_params(target):
        r = np.random.rand(3)
        return r * np.array([target.shape[0], target.shape[1], min(target.shape[:2])])

class Polygon(ShapePrimitive):
    def rasterize(self, shape):
        x,y,r,th = self.params
        angles = np.linspace(0, 2*np.pi, num=self.sides, endpoint=False)
        p =  np.array([ r * np.cos(angles), r * np.sin(angles) ])

        if th != 0:            
            p = np.dot(rot2d(th), p)

        return polygon(p[0,:]+x, p[1,:]+y, shape=shape)

    def scale(self, f):
        x,y,r,th = self.params
        return self.__class__([x*f,y*f,r*f,th], self.alpha, self.color)

    @staticmethod
    def random_params(target):
        r = np.random.rand(4)
        w = np.min(target.shape[:2])
        return r * np.array([ target.shape[0], target.shape[1], w*0.5, 2*np.pi ])

class Triangle(Polygon):
    sides = 3
    
class Hexagon(Polygon):
    sides = 6

class PrimitiveFactory(object):
    PRIMITIVES = { 'ellipse': Ellipse, 
                   'rotated_rectangle': RotatedRectangle, 
                   'rectangle': Rectangle, 
                   'circle': Circle, 
                   'triangle': Triangle, 
                   'hexagon': Hexagon, 
                   'cosine': Cosine, 
                   'gabor': Gabor,
                   'line': Line,
                   'bezier': Bezier,
                   'gaussian': Gaussian,
                   'crescent': Crescent }

    @staticmethod
    def random(ptype, target, **kwargs):        
        return PrimitiveFactory.PRIMITIVES[ptype].random(target, **kwargs)

    @staticmethod
    def new(ptype, params, *args, **kwargs):
        return PrimitiveFactory.PRIMITIVES[ptype](params, *args, **kwargs)
