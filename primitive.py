from skimage.draw import ellipse, polygon, circle
import numpy as np

ELLIPSE = 'ellipse'
ROTATED_RECTANGLE = 'rotated_rectangle'
RECTANGLE = 'rectangle'
CIRCLE = 'circle'
TRIANGLE = 'triangle'
HEXAGON = 'hexagon'

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
    PRIMITIVES = { ELLIPSE: Ellipse, ROTATED_RECTANGLE: RotatedRectangle, RECTANGLE: Rectangle, CIRCLE: Circle, TRIANGLE: Triangle, HEXAGON: Hexagon }

    @staticmethod
    def random(ptype, target):
        return PrimitiveFactory.PRIMITIVES[ptype].random(target)

    @staticmethod
    def new(ptype, params):
        return PrimitiveFactory.PRIMITIVES[ptype](params)
