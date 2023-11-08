import numpy as np
import numpy.typing as npt
from encoder.encoder import Encoder

class PCAEncoder(Encoder):

    # vectors: List of numpy arrays  
    def __init__(self, vectors: npt.NDArray[np.float64], ndim: int =50) -> npt.NDArray[np.float64]:
        super().__init__()

        cov = np.cov(vectors.T)
        # eigen values, and normalized eigenvectors
        evals, evecs = np.linalg.eig(cov)
        p = evals.argsort()[::-1] # descending order
        evals = evals[p]
        evecs = evecs[p]
        self.evals = evals[:ndim]
        self.pcs = evecs[:ndim]        
    
    # project vectors using principal components
    def project(self, vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # project vectors into lower dimension using principal components
        # we don't have to renormailze because pcs are already normalized
        return vectors @ self.pcs.T

 

                    
