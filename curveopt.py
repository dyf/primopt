import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import spline
from scipy.optimize import minimize
import numpy as np
from scipy.misc import imread,imsave
from skimage.draw import line
from primitive import image_error
import os
import skimage.feature
import skimage.color
import scipy.ndimage.morphology

def line_clipped(r0, c0, r1, c1, shape):
    ln = line(r0, c0, r1, c1)
    lngood = [ (ln[i] >= 0) & (ln[i] < shape[i]) for i in range(len(shape)) ]
    lngood = lngood[0] & lngood[1]
    return [ v[lngood] for v in ln ]    

class SplinePrimitive(object):
    def __init__(self, dims=2):
        self.ps = []
        self.vs = []
        self.coeffs = []
        self.dims = 2

    def add_point(self, p, v):
        self.ps.append(p)
        self.vs.append(v)
        return
        if len(self.ps) > 1:
            self.coeffs.append(spline.cubic_spline_coeffs(self.ps[-2], self.vs[-2],
                                                          self.ps[-1], self.vs[-1]))

    def set_point(self, i, p, v):
        self.ps[i] = p
        self.vs[i] = v

        return
        if i < 0:
            i += len(self.ps)
            
        if i > 0:
            self.coeffs[i-1] = spline.cubic_spline_coeffs(self.ps[i-1], self.vs[i-1],
                                                          self.ps[i], self.vs[i])
        if i < (len(self.ps) - 1):
            self.coeffs[i] = spline.cubic_spline_coeffs(self.ps[i], self.vs[i],
                                                        self.ps[i+1], self.vs[i+1])
                                                        
    def add_random_point(self):
        s = np.random.choice([-1,1], size=(self.dims,))
        self.add_point(np.random.random(self.dims),
                       s*np.sqrt(np.random.random(self.dims)))

        
    def randomize_endpoint(self):
        s = np.random.choice([-1,1], size=(self.dims,))
        self.set_point(-1,
                       np.random.random(self.dims),
                       s*np.sqrt(np.random.random(self.dims)))

    def mutate_endpoint(self, d):
        self.set_point(-1,
                       np.random.randn(self.dims)*d + 1,
                       np.random.randn(self.dims)*d + 1)

    def remove_endpoint(self):
        del self.ps[-1]
        del self.vs[-1]

        return
        if len(self.coeffs):
            del self.coeffs[-1]

    def render(self, im, segs=20):
        import scipy.interpolate

        t0 = np.linspace(0, 1, len(self.ps))
        t1 = np.linspace(0, 1, len(self.ps)*segs)

        try:
            xx = np.array([ scipy.interpolate.interp1d(t0,  [ p[0] for p in self.ps ], kind='cubic')(t1),
                            scipy.interpolate.interp1d(t0,  [ p[1] for p in self.ps ], kind='cubic')(t1) ])
        except Exception as e:
            print(self.ps)
            raise
        
        # xx = spline.cubic_spline(segs, coeffs_list=self.coeffs)
        for i in range(self.dims):
            xx[i] *= im.shape[i]
        xx = xx.astype(int)

        im.fill(0)
        for i in range(xx.shape[1]-1):
            ln = line_clipped(xx[0,i],xx[1,i],xx[0,i+1],xx[1,i+1],im.shape)
            im[ln] = 1
        return im
    
    def mutate(self, d):
        spl = SplinePrimitive()
        for i in range(len(self.ps)):
            spl.add_point(self.ps[i] * (np.random.randn(self.dims)*d + 1),
                          self.vs[i] * (np.random.randn(self.dims)*d + 1))
        return spl
                               
    @classmethod
    def random(dims=2):
        spl = SplinePrimitive(dims)
        spl.add_random_point()
        spl.add_random_point()
        spl.add_random_point()
        spl.add_random_point()
        return spl

def gradient_image(im, sqr=False, norm_pct=95):
    im = im.astype(float)
    if len(im.shape) == 2:
        gx, gy = np.gradient(im)
    else:
        gx, gy = np.gradient(im, axis=[0,1])
        gx = gx.max(axis=2)
        gy = gy.max(axis=2)

    gmag = gx*gx + gy*gy

    if sqr:
        gmag = np.sqrt(gmag)
        
    if norm_pct is not None:
        v = np.percentile(gmag[:], 95)
        return gmag / v
    else:
        return gmag

def canny_dist_image(im, sigma):
    gim = skimage.feature.canny(skimage.color.rgb2gray(im).astype(float), sigma=sigma).astype(float)
    gim = scipy.ndimage.morphology.distance_transform_edt(1.0-gim)
    return 1.0 / (gim + 1.0)

def curveopt(im, N_pts=100, N_init=1000, N_rand=1000):
    #gim = gradient_image(im)
    gim = canny_dist_image(im, 2)
    buf = np.zeros_like(gim)
    yield gim

    best_spl = None
    best_error = float("inf")#image_error(buf,gim)
                               
    # pick a good starting point
    for i in range(N_init):
        spl = SplinePrimitive.random()
        spl.render(buf)
        err = image_error(buf,gim)

        if err < best_error:
            best_error = err
            best_spl = spl

    for i in range(N_rand):
        new_spl = best_spl.mutate(.1)
        new_spl.render(buf)
        err = image_error(buf,gim)
                               
        if err < best_error:
            best_error = err
            best_spl = new_spl
                               
    best_spl.render(buf)
    best_error = image_error(buf,gim)
    
    yield buf
                               
    # add new points
    for i in range(N_pts):
        best_p = None
        best_v = None
        
        best_spl.add_random_point()
    
        for j in range(N_init):
            best_spl.randomize_endpoint()
            best_spl.render(buf)
            err = image_error(buf,gim)

            if err < best_error:
                best_error = err
                best_p = best_spl.ps[-1].copy()
                best_v = best_spl.vs[-1].copy()
                               
        for j in range(N_rand):
            best_spl.mutate_endpoint(.1)
            best_spl.render(buf)
            err = image_error(buf,gim)

            if err < best_error:
                best_error = err
                best_p = best_spl.ps[-1].copy()
                best_v = best_spl.vs[-1].copy()

        if best_p is not None:
            best_spl.set_point(-1, best_p, best_v)
            best_spl.render(buf)
            yield buf
                               
def main():
    im = imread("kermit.jpg")
    savedir = "/mnt/c/Users/davidf/workspace/curveopt/"

    for i, cim in enumerate(curveopt(im, N_pts=200, N_init=5000, N_rand=5000)):
        print(i)
        savepath = os.path.join(savedir, "test_%05d.jpg" % i)
        imsave(savepath, 1.0 - cim.clip(0,1))

if __name__ == "__main__": main()
