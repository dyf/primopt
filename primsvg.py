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
        elif isinstance(prim, primitive.Line):
            x1, y1, x2, y2 = prim.params
            patch = mpatches.Line((y1, x1), (y2, x2))
        elif isinstance(prim, primitive.Rectangle):
            x,y,w,h = prim.params
            patch = mpatches.Rectangle((y, x), h, w)
        elif isinstance(prim, primitive.RotatedRectangle):
            verts = prim.vertices()
            patch = mpatches.Polygon(verts[[1,0],:].T, closed=True)
        elif isinstance(prim, primitive.Circle):
            x,y,r = prim.params
            patch = mpatches.Circle((y, x), r)
        elif isinstance(prim, primitive.Polygon):
            verts = prim.vertices()
            patch = mpatches.Polygon(verts[[1,0],:].T, closed=True)
        else:
            pass

        if patch:
            patch.set_facecolor(prim.color)
            patch.set_alpha(prim.alpha)
            ax.add_patch(patch)

    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
