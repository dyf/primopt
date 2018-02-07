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

def optimize_primitive_sync(target, current, r_its, m_its, prim_type, m_fac, prim_props=None):
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


class PrimitiveOptimizer(object):
    def __init__(self, r_its, m_its, n_prims, prim_type, m_fac=.1, prim_props=None, m_props=None, levels=None, n_procs=None):
        self.r_its = r_its
        self.m_its = m_its
        self.n_prims = n_prims
        self.prim_type = prim_type

        self.m_fac = m_fac
        self.prim_props = prim_props if prim_props else {}
        self.m_props = m_props
        self.levels = sorted(levels)[::-1] if levels else [0]

        init_pool(n_procs)

    def optimize_primitive(self, target, current):
        for pi in range(self.n_prims):               
            current_error = primitive.image_error(current, target)

            if POOL:
                nprocs = POOL._processes
                resps = []
                for i in range(POOL._processes):
                    resp = POOL.apply_async(optimize_primitive_sync, args=(target, current,
                                                                           self.r_its//nprocs, self.m_its//nprocs,
                                                                           self.prim_type, self.m_fac, self.prim_props))
                    resps.append(resp)

                resps = [ r.get() for r in resps ]

                errors = np.array(r[0] for r in resps)

                best_i = np.argmin(errors)
                best_error, best_prim = resps[best_i]
            else:
                best_error, best_prim = optimize_primitive_sync(target, current,
                                                                r_its, m_its,
                                                                prim_type, m_fac, prim_props)

            if best_error > current_error:
                continue 

            try:
                current = best_prim.draw(current, target)
                yield current, best_prim
            except primitive.EmptyPrimitiveException as e:
                pass

    def optimize_level(self, target, current, level):
        if level == 0:
            for c, prim in self.optimize_primitive(target, current):
                yield c, prim
        else:
            scale = 2 ** level
            target_scale = target[::scale, ::scale, :]
            current_scale = current[::scale, ::scale, :]
            
            for _, prim in self.optimize_primitive(target_scale, current_scale):

                prim_s = prim.scale(float(scale))

                try:
                    current = prim_s.draw(current, target)
                    yield current, prim_s
                except primitive.EmptyPrimitiveException as e:
                    pass

    def optimize(self, target, current):
        current = current.copy()

        for level in self.levels:
            for current, prim in self.optimize_level(target, current, level):
                yield current, prim


POOL = None
