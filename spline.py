# f(x) = a*x*x*x + b*x*x + c*x + d
# f'(x) = 3*a*x*x + 2*b*x + c
#
# d = x0
# c = dx0
# a + b + c + d = x1
# 3*a + 2*b + c = dx1
#
# a + b + dx0 + x0 = x1
# a + b = x1 - x0 - dx0 
# a = x1 - x0 - dx0 - b
#
# 3*a + 2*b + dx0 = dx1
# 3*a + 2*b = dx1 - dx0
# 3*(x1 - x0 - dx0 - b) + 2*b = dx1 - dx0
# -3*b + 2*b = dx1 - dx0 - 3*(x1 - x0 - dx0)
# b = -dx1 + dx0 + 3*(x1 - x0 - dx0)
#
# a = x1 - x0 - dx0 - 0.5 * (dx1 - dx0 - 3*(x1 - x0 - dx0))
def cubic_spline_coeffs(ps, vs):
    coeffs = []
    for i in range(N-1):
        p0 = ps[i]
        p1 = ps[i+1]
        v0 = vs[i]
        v1 = vs[i+1]

        d = p0
        c = v0
        b = -v1 + v0 + 3*(p1 - p0 - v0)
        a = p1 - p0 - v0 - b        
        coeffs.append([a,b,c,d])
    
    return np.array(coeffs)

def cubic_spline(ps, vs, N):
    t = np.linspace(0,1,N)[np.newaxis].T
    coeffs_list = cubic_spline_coeffs(ps,vs)        
    vs = []
    for a,b,c,d in coeffs_list:
        v = a*t**3 + b*t**2 + c*t + d
        vs.append(v)
    return np.concatenate(vs).T