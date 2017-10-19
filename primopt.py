from skimage.draw import ellipse
import matplotlib.pyplot as plt
import numpy as np

class Ellipse(object):
    def __init__(self, x, y, r1, r2, rot, color):
        self.x = x
        self.y = y
        self.r1 = r1
        self.r2 = r2
        self.rot = rot
        self.color = color
    
    def render(self, shape):
        rr,cc = ellipse(self.x, self.y, self.r1, self.r2, shape[:2], self.rot)
        im = np.zeros(list(shape)[:2] + [len(self.color)])
        im[rr,cc,:] = self.color
        return im
    
    def mutate(self, d):
        r = 1 + np.random.randn(5 + len(self.color)) * d
        return Ellipse(self.x * r[0],
                       self.y * r[1],
                       self.r1 * r[2],
                       self.r2 * r[3],
                       self.rot * r[4],
                       self.color * r[5:])
    def params(self):        
        return np.concatenate(([ self.x, self.y, self.r1, self.r2, self.rot ], self.color))
        
    @staticmethod
    def from_params(x):
        return Ellipse(x[0], x[1], x[2], x[3], x[4], x[5:])
    
    @staticmethod
    def random(image_shape, color_dims):
        r = np.random.rand(5 + color_dims)
        return Ellipse(r[0]*image_shape[0],
                       r[1]*image_shape[1],
                       r[2]*np.linalg.norm(image_shape[:2])*0.5,
                       r[3]*np.linalg.norm(image_shape[:2])*0.5,
                       2.0 * np.pi * r[4],
                       r[5:])

def optimize_image(target, r_its, m_its, n_prims):    
    current = np.ones_like(target)
    current[:,:,0] = scipy.stats.mode(target[:,:,0], axis=None).mode[0]
    current[:,:,1] = scipy.stats.mode(target[:,:,1], axis=None).mode[0]
    current[:,:,2] = scipy.stats.mode(target[:,:,2], axis=None).mode[0]
    
    for pi in range(n_prims):
        print pi        
        
        shapes = [ Ellipse.random(target.shape, 4) for i in range(r_its) ]
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
        plt.imshow(current)
        plt.show()
        
    return current
    
def blend_image(current, im):
    alpha_im = im[:,:,3]
    return current * (1 - alpha_im)[:,:,np.newaxis] + im[:,:,:3] * alpha_im[:,:,np.newaxis]

def error_function(s, current, target):    
    im = s.render(target.shape)    
    blend = blend_image(current, im)
    error = np.sqrt(((blend - target) ** 2)).mean()
    return error

if __name__ == "__main__":
    import scipy.misc
    im = scipy.misc.imread('pufferfish.jpg')[::2,::2,:].astype(float) / 255.0
    bim = optimize_image(im, 100, 100, 10)
    plt.imshow(bim)
    plt.show()