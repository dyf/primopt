import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import primitive
import numpy as np

def save(init_image, prims, path):
    fig, ax = plt.subplots()
    plt.imshow(init_image)

    for prim in prims:
        patch = None
        
        if isinstance(prim, primitive.Ellipse):
            x, y, r1, r2, rot = prim.params
            patch = mpatches.Ellipse((y,x), r2*2, r1*2, -rot*180.0/np.pi)
            patch.set_facecolor(prim.color)
            patch.set_alpha(prim.alpha)
        else:
            pass

        if patch:
            ax.add_patch(patch)

    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
