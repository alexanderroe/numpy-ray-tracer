from audioop import add
import time
import numpy as np
import sys
from PIL import Image
from Material import *
from Hittable import Sphere

# global vars
scene = []
rd = 10 # max recursion depth

# main raytracing function
def raytrace(width: float, height: float) -> np.ndarray:

    aspect = width / height

    # O is a numpy array of shape (height, width, 3) which contains the origin of the ray for each pixel
    # use np.zeros because camera is at origin
    O = np.zeros(shape=(height, width, 3), dtype=np.float32)

    # D is a numpy array of shape (height, width, 3) which contains the direction of the ray for each pixel
    D = np.empty(shape=(height, width, 3), dtype=np.float32) 

    # with camera at origin and projection plane at z = -1, rays have form (x,y,-1)
    # with x in [-aspect, aspect] and y in [-1,1] (so the height is fixed)
    D[:,:,0] = np.tile(np.linspace(-aspect, aspect, num=width), (height, 1))
    D[:,:,1] = np.tile(np.linspace(1, -1, num=height), (width, 1)).T
    D[:,:,2] = np.full((height, width), -1.0)

    # T holds the best t value for each pixel with intersection    
    T = np.full((height, width), np.inf)

    # Io holds object index for the best intersection
    Io = np.full((height, width), np.NINF)

    # R holds the RGB info for the final image, which we will return
    R = np.full((height,width,3), 1.0)

    # "Relevant" rays, which are still bouncing around in the scene
    Rel = np.full((height, width), True)

    # main loop
    for r in range(rd): 

        print('beginning raytrace iteration {}'.format(r))
        s = time.time()

        # compute intersections
        for idx, obj in enumerate(scene):

            # get intersection times for each relevant ray
            I = obj.intersect(O[Rel], D[Rel])

            # mark better intersections with this obj's index
            Io[Rel] = np.maximum(Io[Rel], np.where(np.logical_and(I != np.inf, I < T[Rel]), idx, np.NINF))

            # record any better intersection times
            T[Rel] = np.minimum(T[Rel], I)


        # for all pixels with no intersection, factor in the background color
        bg = (T == np.inf)
        Db = D[bg]
        y = Db[:,1] / np.einsum('ij,ij->i', Db, Db)
        t = np.clip(0.5 * (y + 1), 0, 2)
        bgx = np.empty((len(t), 3))
        bgx[:,0] = 1 - (0.5 * t)
        bgx[:,1] = 1 - (0.3 * t)
        bgx[:,2] = 1
        R[bg] *= bgx
        T[T == np.inf] = np.NINF


        # if any rays survive all recursion levels
        if r == rd-1:
            left = (T != np.NINF)
            R[left] *= np.zeros(3)
            break

        # compute colors
        for idx, obj in enumerate(scene):

            # get pixels corresponding to this obj
            mask = (Io == idx)
            
            # update ray origins to new scattered ray origins
            O[mask] += (D[mask].T * T[mask]).T

            # color using this object's material
            obj.material.scatter(mask, R, D, obj.compute_normals(O[mask]), O, T)

        # update relevant rays
        Rel = (Io != np.NINF)

        # reset T, Io arrays
        T[T != np.NINF] = np.inf
        Io.fill(np.NINF)

        e = time.time()
        print('finished raytrace iteration {} in {} seconds'.format(r, e-s))
        
    return R

# initialize the scene in here
def init_scene() -> None:

    # helper function
    add_sphere = lambda cen, rad, mat: scene.append(Sphere(cen, rad, mat))

    # ground
    add_sphere([0, -100001, 0], 100000, CheckerMetalMaterial([0.7,0.7,0.7]))

    # spheres
    add_sphere([0, 0, -3], 1, DiffuseMaterial([0.1,0.2,0.5]))
    add_sphere([-7, 3, -7], 3, MetalMaterial([0.7,0.7,0.7]))
    add_sphere([0, 2, -7], 3, MetalMaterial([0.9,0.9,0.9]))
    add_sphere([7, 3, -7], 3, MetalMaterial([1,1,1]))


    add_sphere([-3, 0, -4], 1, DiffuseMaterial([0.5,.7,.2]))
    add_sphere([3, 0, -4], 1, DiffuseMaterial([0.1,.7,.9]))

    add_sphere([-7, 0, -4], 1, DiffuseMaterial([1,.7,.2]))
    add_sphere([7, 0, -4], 1, DiffuseMaterial([0.8,0,.1]))

    add_sphere([-2, -0.75, -2], 0.25, LightMaterial([1,1,1]))
    add_sphere([0, -0.75, -2], 0.25, LightMaterial([1,1,0.75]))
    add_sphere([2, -0.75, -2], 0.25, LightMaterial([1,1,1]))


if __name__ == '__main__':
    
    # command line stuff
    if len(sys.argv) != 4:
        print('usage: python3 rt.py <width> <height> <spp>')
        sys.exit()

    if not (sys.argv[1].isdigit() and sys.argv[2].isdigit() and sys.argv[3].isdigit()):
        print('all arguments must be integers')
        sys.exit()
        
    aa = int(sys.argv[3])
    width, height = aa * int(sys.argv[1]), aa * int(sys.argv[2])

    init_scene()

    print('beginning raytracing...')
    s = time.time()
    img = raytrace(width, height)
    e = time.time()
    print('done, took {} seconds'.format(e-s))

    # averaging/downscaling for supersampling anti-aliasing
    img = img.reshape(height // aa, aa, width // aa, aa, 3).sum((1,3)) / (aa ** 2)

    # convert and save image
    nimg = Image.fromarray(np.uint8(np.round(np.clip(img, 0, 0.999) * 255)), 'RGB')
    nimg.save('render.png')