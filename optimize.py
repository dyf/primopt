import os
from scipy.ndimage.interpolation import zoom
import numpy as np
import scipy.stats
import primitive as primitive
import time
import multiprocessing as mp



def init_pool(procs=None):    
    global POOL
    if procs is None:
        procs = mp.cpu_count()
        
    if procs > 1:
        POOL = mp.Pool(processes=procs)
    else:
        POOL = None


def optimize_image_levels(target, r_its, m_its, n_prims, levels, prim_type, current):
    if current is None:
        current = mean_image(target)
    else:
        current = current.copy()

    pi = 1
    for level in range(levels,0,-1):
        f = 2 ** level
        target_level = target[::f,::f,:].copy()
        current_level = current[::f,::f,:].copy()
        #f = float(target.shape[0]) / float(target_level.shape[0]) 
        
        failed_images = 0

        current_error = primitive.image_error(current, target)

        for cim, prim, i in optimize_image(target_level, r_its, m_its, n_prims, prim_type, current_level):
            scale_prim = prim.scale(float(f))

            next_im = scale_prim.draw(current, target)
            error = primitive.image_error(next_im, target)

            if error > current_error:
                failed_images += 1
                if failed_images > 20:
                    print("stopping with level, too many failed images")
                    break

                continue
            
            current_error = error
            np.copyto(current,next_im)

            yield current, scale_prim, pi
            pi += 1

        print("finished level %d" % level)

    for cim, prim, i in optimize_image(target, r_its, m_its, n_prims, prim_type, current):
        yield cim, prim, pi
        pi += 1


def optimize_image_primitive(target, current, r_its, m_its, prim_type, m_fac, prim_props=None):
    np.random.seed(seed=int(time.time())+os.getpid())

    prim_props = prim_props if prim_props else {}
    
    best_error = float("inf")
    best_prim = None

    for i in range(r_its):
        prim = primitive.PrimitiveFactory.random(prim_type, target, **prim_props)
        error = prim.error(current, target)        
        
        if error < best_error:
            best_prim = prim
            best_error = error
    
                
    for i in range(m_its):
        next_prim = best_prim.mutate(m_fac)
        error = next_prim.error(current, target)
        if error < best_error:
            best_prim = next_prim
            best_error = error

    return (best_error, best_prim)


def optimize_image(target, r_its, m_its, n_prims, prim_type, current=None, m_fac=.1, prim_props=None, m_props=None):
    if current is None:
        current = mean_image(target)
    else:
        current = current.copy()

    for pi in range(n_prims):               
        current_error = primitive.image_error(current, target)

        if POOL:
            nprocs = POOL._processes
            resps = []
            for i in range(POOL._processes):
                resp = POOL.apply_async(optimize_image_primitive, args=(target, current, r_its//nprocs, m_its//nprocs, prim_type, m_fac, prim_props))
                resps.append(resp)

            resps = [ r.get() for r in resps ]

            errors = np.array(r[0] for r in resps)

            best_i = np.argmin(errors)
            best_error, best_prim = resps[best_i]
        else:
            best_error, best_prim = optimize_image_primitive(target, current, r_its, m_its, prim_type, m_fac, prim_props)

        if best_error > current_error:
            continue 

        current = best_prim.draw(current, target)
        yield current, best_prim, pi+1

def mode_image(im):
    out = np.ones_like(im)
    out[:,:,0] = scipy.stats.mode(im[:,:,0], axis=None).mode[0]
    out[:,:,1] = scipy.stats.mode(im[:,:,1], axis=None).mode[0]
    out[:,:,2] = scipy.stats.mode(im[:,:,2], axis=None).mode[0]
    return out

def mean_image(im):
    out = np.ones_like(im)
    out[:,:,:] = im.mean(axis=(0,1))
    return out

POOL = None
