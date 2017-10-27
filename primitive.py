from skimage.draw import ellipse, polygon, circle, line
from skimage.filters import gabor_kernel
import numpy as np

ELLIPSE = 'ellipse'
ROTATED_RECTANGLE = 'rotated_rectangle'
RECTANGLE = 'rectangle'
CIRCLE = 'circle'
TRIANGLE = 'triangle'
HEXAGON = 'hexagon'
COSINE = 'cosine'
GABOR = 'gabor'
GAUSSIAN = 'gaussian'
LINE = 'line'

ADD = 'add'
COMPOSITE = 'composite'

def image_error(image, target):    
    return ((image - target) ** 2).mean()

def rot2d(th):
    return np.array([[ np.cos(th), -np.sin(th)], 
                     [ np.sin(th), np.cos(th) ]])

class Primitive(object):
    def __init__(self, params, alpha, color=None):
        self.params = np.array(params)
        self.alpha = float(alpha)
        self.color = color        
        self.mode = COMPOSITE

    def error(self, current, target):
        blend = self.draw(current, target)
        return image_error(blend, target)
        
    def select_color(self, current, target, px):
        if len(px[0]) == 0:
            return None
        idx = np.random.choice(len(px[0]))
        return target[px[0][idx], px[1][idx], :]

    def mutate(self, d):
        r = 1 + np.random.randn(len(self.params)+1) * d
        return self.__class__(r[:-1] * self.params, self.alpha * r[-1])



class ShapePrimitive(Primitive):       

    def draw(self, current, target):
        
        mask_px = self.rasterize(target.shape)

        out = current.copy()

        if self.color is None:
            self.color = self.select_color(current, target, mask_px)
        
        if self.color is None:
            return out
        
        if self.mode == COMPOSITE:
            out[mask_px[0], mask_px[1], :] = out[mask_px[0], mask_px[1], :] * (1.0 - self.alpha) + self.color * self.alpha
        elif self.mode == ADD:
            out[mask_px[0], mask_px[1], :] += self.color * self.alpha
        return out

class ImagePrimitive(Primitive):
    def draw(self, current, target):
        image = self.rasterize(target.shape)

        if self.color is None:
            self.color = self.select_color(current, target, np.where(image != 0))
        if self.color is None:
            return current.copy()

        if self.mode == ADD:
            mix = current + image[:,:,np.newaxis] * self.color * self.alpha
        elif self.mode == COMPOSITE:
            mix = current * (1.0 - self.alpha) + image[:,:,np.newaxis] * self.color * self.alpha
        
        
        return mix

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
    def random(target):
        r = np.random.rand(7)
        return Gabor([np.random.randint(2,target.shape[0]-2), np.random.randint(2,target.shape[1]-2),
                      r[0] + (1.0 - r[0]) / target.shape[0], r[1]*np.pi, 
                      r[2]*target.shape[0]], r[3], r[4:])

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
    def random(target):
        r = np.random.rand(4)
        return Cosine([r[0]*target.shape[0], r[1]*target.shape[1], r[2]*target.shape[0], r[3]*target.shape[1]], 1.0)

    def mutate(self, d):
        r = np.random.randn(4) * d
        return self.__class__(r * self.params, 1.0)
                      

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
    def random(target):
        r = np.random.rand(5)
        return Line(r[:4] * np.array([ target.shape[0]-1, target.shape[1]-1, target.shape[0]-1, target.shape[1]-1]), r[-1])

class Ellipse(ShapePrimitive):
    def rasterize(self, shape):
        x, y, r1, r2, rot = self.params
        return ellipse(x, y, r1, r2, shape[:2], rot)

    def scale(self, f):
        x, y, r1, r2, rot = self.params
        return Ellipse([x*f, y*f, r1*f, r2*f, rot], self.alpha)

    @staticmethod
    def random(target):
        w = min(target.shape[:2])
        r = np.random.rand(5)
        return Ellipse(r * np.array([ target.shape[0], target.shape[1], (w+1)*0.5, (w+1)*0.5, 2.0*np.pi]), 
                       np.random.rand())

class Rectangle(ShapePrimitive):
    def rasterize(self, shape):
        x,y,w,h = self.params
        p = np.array([ [x, x, x+w, x+w], [y, y+h, y+h, y] ])
        return polygon(p[0,:], p[1,:], shape=shape)

    def scale(self, f):
        x,y,w,h = self.params
        return Rectangle([x*f, y*f, w*f, h*f], self.alpha, self.color)

    @staticmethod
    def random(target):
        r = np.random.rand(4)
        return Rectangle(r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1]),
                         np.random.rand())

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
    def random(target):
        r = np.random.rand(5)
        return RotatedRectangle(r * np.array([target.shape[0], target.shape[1], target.shape[0]+1, target.shape[1]+1, 2.0*np.pi]),
                                np.random.rand())
class Circle(ShapePrimitive):
    def rasterize(self, shape):
        x,y,r = self.params
        return circle(x, y, r, shape=shape)

    def scale(self, f):
        x,y,r = self.params
        return Circle([x*f,y*f,r*f], self.alpha, self.color)

    @staticmethod
    def random(target):
        r = np.random.rand(3)
        return Circle(r * np.array([target.shape[0], target.shape[1], min(target.shape[:2])]),
                      np.random.rand())

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

    @classmethod
    def random(cls, target):
        r = np.random.rand(4)
        w = np.min(target.shape[:2])
        return cls(r * np.array([ target.shape[0], target.shape[1], w*0.5, 2*np.pi ]),
                   np.random.rand())

class Triangle(Polygon):
    sides = 3
    
class Hexagon(Polygon):
    sides = 6

class PrimitiveFactory(object):
    PRIMITIVES = { ELLIPSE: Ellipse, 
                   ROTATED_RECTANGLE: RotatedRectangle, 
                   RECTANGLE: Rectangle, 
                   CIRCLE: Circle, 
                   TRIANGLE: Triangle, 
                   HEXAGON: Hexagon, 
                   COSINE: Cosine, 
                   GABOR: Gabor,
                   LINE: Line }

    @staticmethod
    def random(ptype, target):
        return PrimitiveFactory.PRIMITIVES[ptype].random(target)

    @staticmethod
    def new(ptype, params, alpha, *args, **kwargs):
        return PrimitiveFactory.PRIMITIVES[ptype](params, alpha, *args, **kwargs)
