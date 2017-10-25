from skimage.draw import ellipse, polygon, circle
import numpy as np

ELLIPSE = 'ellipse'
ROTATED_RECTANGLE = 'rotated_rectangle'
RECTANGLE = 'rectangle'
CIRCLE = 'circle'
TRIANGLE = 'triangle'
HEXAGON = 'hexagon'
COSINE = 'cosine'

def image_error(image, target):    
    return ((image - target) ** 2).mean()

def rot2d(th):
    return np.array([[ np.cos(th), -np.sin(th)], 
                     [ np.sin(th), np.cos(th) ]])

class Primitive(object):
    def __init__(self, params, alpha):
        self.params = np.array(params)
        self.alpha = float(alpha)
        
    def error(self, current, target):
        blend = self.draw(current, target)

        return image_error(blend, target)

    def mutate(self, d):
        r = 1 + np.random.randn(len(self.params)+1) * d
        return self.__class__(r[:-1] * self.params, self.alpha * r[-1])

class ImagePrimitive(Primitive):
    def __init__(self, params, alpha):
        super(ImagePrimitive, self).__init__(params, alpha)        
        self.channel = None

    def draw(self, current, target):
        image = self.rasterize(target.shape)

        out = current.copy()
        
        if self.channel is None:
            self.channel = np.random.randint(0,3)

        out[:,:,self.channel] += image * self.alpha

        return out

class Cosine(ImagePrimitive):
    def rasterize(self, shape):
        sfx, sfy = self.params        
        xx,yy = np.mgrid[0:shape[0],0:shape[1]]

        return (np.cos(2*np.pi*xx*sfx) + np.cos(2*np.pi*yy*sfy))* 0.5 + 1

    def scale(self, f):
        sfx, sfy = self.params
        return Cosine([sfx*f, sfy*f], self.alpha)

    @staticmethod
    def random(target):
        r = np.random.rand(3)
        return Cosine([ 1.0 / (target.shape[0] * r[0]), 1.0 / (target.shape[1] * r[1]) ], r[2]*2.0 - 1.0)
                      

class ShapePrimitive(Primitive):
    def __init__(self, params, alpha, color=None):
        super(ShapePrimitive, self).__init__(params, alpha)        
        self.color = color
        
    def select_color(self, current, target, mask_px):
        idx = np.random.choice(len(mask_px[0]))
        return target[mask_px[0][idx], mask_px[1][idx], :].copy()

    def draw(self, current, target):
        mask_px = self.rasterize(target.shape)

        out = current.copy()

        if len(mask_px[0]) == 0:
            return out
        
        if self.color is None:
            self.color = self.select_color(current, target, mask_px)

        out[mask_px[0], mask_px[1], :] = out[mask_px[0], mask_px[1], :] * (1.0 - self.alpha) + self.color * self.alpha

        return out

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
    PRIMITIVES = { ELLIPSE: Ellipse, ROTATED_RECTANGLE: RotatedRectangle, RECTANGLE: Rectangle, CIRCLE: Circle, TRIANGLE: Triangle, HEXAGON: Hexagon, COSINE: Cosine }

    @staticmethod
    def random(ptype, target):
        return PrimitiveFactory.PRIMITIVES[ptype].random(target)

    @staticmethod
    def new(ptype, params, alpha, *args, **kwargs):
        return PrimitiveFactory.PRIMITIVES[ptype](params, alpha, *args, **kwargs)
