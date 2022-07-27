from abc import ABC, abstractmethod
from Material import Material
import numpy as np

# abstract base class for hittable objects
class Hittable(ABC):

    @abstractmethod
    def intersect(O: np.ndarray, D: np.ndarray) -> np.ndarray:
        pass

# sphere "extends" Hittable 
class Sphere(Hittable):

    def __init__(self, center, radius: float, material: Material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def compute_normals(self, O):
        return (O - self.center) / self.radius

    # given ray origins and directions O and D,
    # return a boolean mask of which rays intersect the AABB
    def hit_aabb(self, O, D):

        l = self.center - self.radius
        h = self.center + self.radius

        tmin = np.zeros((O.shape[0],))
        tmax = np.full((O.shape[0],), np.inf)

        s0 = (l[0] - O[:,0]) / D[:,0]
        s1 = (h[0] - O[:,0]) / D[:,0]
        t0 = np.minimum(s0, s1)
        t1 = np.maximum(s0, s1)

        tmin = np.maximum(tmin, t0)
        tmax = np.minimum(tmax, t1)

        s2 = (l[1] - O[:,1]) / D[:,1]
        s3 = (h[1] - O[:,1]) / D[:,1]
        t2 = np.minimum(s2, s3)
        t3 = np.maximum(s2, s3)

        tmin = np.maximum(tmin, t2)
        tmax = np.minimum(tmax, t3)

        s4 = (l[2] - O[:,2]) / D[:,2]
        s5 = (h[2] - O[:,2]) / D[:,2]
        t4 = np.minimum(s4, s5)
        t5 = np.maximum(s4, s5)

        tmin = np.maximum(tmin, t4)
        tmax = np.minimum(tmax, t5)

        return (tmin <= tmax)


    # given ray origins and directions O and D, return an (h,w) array I where
    # I == np.inf if no intersection, otherwise I is the distance to the intersection point
    def intersect(self, O: np.ndarray, D: np.ndarray):

        H = self.hit_aabb(O, D)
        O = O[H]
        D = D[H]

        # using quadratic formula, compute determinants
        oc = O - self.center
        a = np.einsum('ij,ij->i', D, D)
        h = np.einsum('ij,ij->i', oc, D)
        c = np.einsum('ij,ij->i', oc, oc) - (self.radius ** 2)
        disc = np.power(h, 2) - np.multiply(a,c)

        # filter out rays with no intersection to avoid unnecessary x-value calcs
        solns = (disc >= 0)

        # compute minimum nonnegative t value
        x1 = (np.negative(h[solns]) - np.sqrt(disc[solns])) / a[solns]
        x2 = (np.negative(h[solns]) + np.sqrt(disc[solns])) / a[solns]

        # assuming all objects are placed in front of camera, no ray intersections will return negative t value. so simply take min of x1, x2 solns
        np.place(disc, disc >= 0, np.minimum(x1, x2))

        # set negative t values to inf
        disc[disc < 0] = np.inf

        I = np.where(H == True, 0, np.inf)
        np.place(I, I == 0, disc)
        return I