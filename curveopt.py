import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from spline import cubic_spline
from scipy.optimize import minimize
import numpy as np
from scipy.misc import imread,imsave
from skimage.draw import line

def line_clipped(r0, c0, r1, c1, shape):
    ln = line(r0, c0, r1, c1)
    lngood = [ (ln[i] >= 0) & (ln[i] < shape[i]) for i in range(len(shape)) ]
    lngood = lngood[0] & lngood[1]
    return [ v[lngood] for v in ln ]    

class SplinePrimitive(object):
    def __init__(self, dims=2):
        self.ps = []
        self.vs = []
        self.dims = 2

    def add_random_point(self):
        self.ps.append(np.random.random(self.dims))
        self.vs.append(np.random.random(self.dims)*2-1)

    def mutate(self, d):
        self.ps[-1] *= (np.random.randn(2)*d + 1)
        self.vs[-1] *= (np.random.randn(2)*d + 1)

    def render(self, im, segs=100):
        xx = cubic_spline(self.ps, self.vs, segs)
        for i in range(self.dims):
            xx[i] *= im.shape[i]
        xx = xx.astype(int)

        im.fill(0)
        for i in range(xx.shape[1]-1):
            ln = line_clipped(xx[0,i],xx[1,i],xx[0,i+1],xx[1,i+1],im.shape)
            im[ln] = 1
        return im
        
def main():
    im = np.random.random((100,100))
    gx, gy = np.gradient(im)
    gim = np.sqrt(gx*gx + gy*gy)

    im = np.zeros((100,100))
    
    sp = SplinePrimitive()
    sp.add_random_point()
    sp.add_random_point()
    sp.render(im)
    imsave('test0.jpg',im)
    sp.add_random_point()
    sp.render(im)
    imsave('test1.jpg',im)
    sp.mutate(.1)
    sp.render(im)
    imsave('test2.jpg',im)
    sp.mutate(.1)
    sp.render(im)
    imsave('test3.jpg',im)
    sp.mutate(.1)
    sp.render(im)
    imsave('test4.jpg',im)

    

    

if __name__ == "__main__": main()
