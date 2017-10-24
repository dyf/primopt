# primopt

PRimitive IMage OPtimizer

Compose an image from random primitives (circles, triangles, ellipses, etc).  

Or: personal challenge to see how quickly I could hack something like https://github.com/fogleman/primitive together in python.

### Elliptical Swedish Chef
![Swedish Chef in Ellipses](https://github.com/dyf/primopt/blob/master/examples/swedish.png)

### Triangular Beaker
![Beaker in Triangles](https://github.com/dyf/primopt/blob/master/examples/beaker.png)

### Hexagonal Animal
![Animal in Hexagons](https://github.com/dyf/primopt/blob/master/examples/animal.png)

### Pseudocode
1) Pick `M` random circles (position, size) with random alpha and random color chosen from input image
2) Pick the circle that has minimal RMSE error when blended into current canvas
3) Stochastic hillclimb: mutate that circle `N` times, choosing new circle if it has lower error
4) Blend best shape into canvas
5) Repeat
6) Optionally: start 1-5 at an aggressively subsampled image resolution and use low-res primitives to seed the higher resolutions
