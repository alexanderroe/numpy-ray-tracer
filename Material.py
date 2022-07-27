from abc import ABC, abstractmethod
import numpy as np

# abstract base class for materials
class Material(ABC):

    @abstractmethod
    def scatter(self, mask: np.ndarray, R: np.ndarray, D: np.ndarray, N: np.ndarray, O: np.ndarray, T: np.ndarray) -> None:
        pass

class DiffuseMaterial(Material):

    def __init__(self, albedo):
        self.albedo = np.array(albedo)

    def scatter(self, mask, R, D, N, O, T):
        
        # random unit vectors for diffuse reflection
        v = np.random.rand(N.shape[0], 3) - 0.5
        v = v / np.linalg.norm(v, axis=1, keepdims=True)

        # update scattered ray directions
        D[mask] = N + v

        # update R with albedo
        R[mask] *= self.albedo 


class MetalMaterial(Material):
    
    def __init__(self, albedo):
        self.albedo = np.array(albedo)

    def scatter(self, mask, R, D, N, O, T):

        # unitize D
        D[mask] /= np.linalg.norm(D[mask], axis=1, keepdims=True)
        
        # reflect D across N
        D[mask] -= 2 * np.einsum('ij,ij->i', D[mask], N).reshape(-1,1) * N
        
        # update R with albedo
        R[mask] *= self.albedo

class DielectricMaterial(Material):

    def __init__(self, refraction_index):
        self.refraction_index = refraction_index

    def scatter(self, mask, R, D, N, O, T):
        
        # calculate front facing normals to determine refraction ratios
        ff = np.einsum('ij,ij->i', D[mask], N)
        ref_ratio = np.where(ff > 0, self.refraction_index, 1.0 / self.refraction_index).reshape(-1,1)
        
        # some trigonometry
        ud = D[mask] / np.linalg.norm(D[mask], axis=1, keepdims=True)
        cos_theta = np.minimum(1.0, np.einsum('ij,ij->i', np.negative(ud), N))
        
        # calculate refracted ray directions
        r_out_perp = ref_ratio * (ud + cos_theta.reshape(-1,1) * N)
        r_out_parallel = np.sqrt(np.abs(1.0 - np.einsum('ij,ij->i', r_out_perp, r_out_perp))).reshape(-1,1) * N

        D[mask] = r_out_perp + r_out_parallel

class LightMaterial(Material):

    def __init__(self, emit):
        self.emit = np.array(emit)

    def scatter(self, mask, R, D, N, O, T):
        R[mask] = self.emit

        # do not scatter rays
        T[mask] = np.NINF


class CheckerMetalMaterial(Material):
    def __init__(self, albedo):
        self.albedo = np.array(albedo)

    def scatter(self, mask, R, D, N, O, T):

        # NOTE: this is a bit of a hack to get the checker pattern to work.
        # the idea is to first convert a point p on an object's surface into integer coords by rounding down.
        # then, if the sum of the x + y + z coords is even, the point is on the dark side of the checker.
        # otherwise, it is on the light side. This creates a pseudo-checker pattern without any uv mapping stuff.

        # round to nearest integer
        Om = np.floor(O[mask])

        # now, if Om is even, then color with albedo, otherwise don't
        Om = Om[:,0] + Om[:,1] + Om[:,2]
        checker = (Om % 2 == 0)

        # update R with albedo and checker
        R[mask] *= self.albedo
        R[mask] = np.multiply(R[mask].T, checker).T

        # unitize D
        D[mask] /= np.linalg.norm(D[mask], axis=1, keepdims=True)
        
        # reflect D across N
        D[mask] -= 2 * np.einsum('ij,ij->i', D[mask], N).reshape(-1,1) * N