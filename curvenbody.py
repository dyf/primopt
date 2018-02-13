import numpy as np
import imageio
import skimage.feature
import scipy

def canny_dist_image(im, sigma):
    gim = skimage.feature.canny(skimage.color.rgb2gray(im).astype(float), sigma=sigma).astype(float)
    gim = scipy.ndimage.morphology.distance_transform_edt(1.0-gim)
    return 1.0 / (gim + 1.0)

def curve_nbody(img, dt=0.1, G=1.0, m=1.0, sigma=2):
    dist_img = canny_dist_image(img, sigma)

    px = np.array(np.where(dist_img > 0))
    weights = dist_img[px[0],px[1]]
    pos = np.array([ img.shape[0] * 0.5, img.shape[1] * 0.5 ])
    v = np.array([0.,0.])
    print(v.shape)
    
    while(True):
        f = px - np.array([pos]).T
        r = np.linalg.norm(f, axis=0)
        f /= r
        f = (G * f * weights / (r**2)).sum(axis=1)

        dv = f * dt
        dp = dv * dt

        v += dv
        pos += dp
        print(pos)

def main():
    img = imageio.imread('kermit.jpg')
    curve_nbody(img)

    
    
    

if __name__ == "__main__": main()
