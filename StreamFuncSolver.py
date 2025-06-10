import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import solve_banded, lstsq

class StreamFuncSolOnLatLon:
    """
    Fast Poisson solver on a latitude-longitude patch using:
      - DCT-II in longitude (Neumann BCs)
      - Tridiagonal solves in latitude
    Also computes vorticity and velocity from streamfunction.
    """
    def __init__(self, lat: np.ndarray, lon: np.ndarray, a_radius: float = 6.371e6):
        # Earth's radius and grid
        self.a = a_radius
        self.φ = np.deg2rad(lat)
        self.λ = np.deg2rad(lon)
        self.Nφ, self.Nλ = self.φ.size, self.λ.size
        self.Δφ = self.φ[1] - self.φ[0]
        self.Δλ = self.λ[1] - self.λ[0]

        # Precompute metric terms
        self.cosφ = np.cos(self.φ)
        self.cosφ_p = np.cos(self.φ + 0.5*self.Δφ)
        self.cosφ_m = np.cos(self.φ - 0.5*self.Δφ)
        # Precompute φ-band coefficients
        self._build_lat_bands()
        # Precompute λ eigenvalues for DCT-II Neumann
        k = np.arange(self.Nλ)
        self._lam_eig = -4.0 * np.sin(np.pi*k/(2*(self.Nλ-1)))**2 / (self.Δλ**2)

    def _build_lat_bands(self):
        Nφ = self.Nφ
        B = np.zeros(Nφ)
        C = np.zeros(Nφ)
        D = np.zeros(Nφ)
        for j in range(Nφ):
            if j == 0:
                Bj = 0.0
                Dj = self.cosφ[j]*self.cosφ_p[j]/self.Δφ**2
            elif j == Nφ-1:
                Bj = self.cosφ[j]*self.cosφ_m[j]/self.Δφ**2
                Dj = 0.0
            else:
                Bj = self.cosφ[j]*self.cosφ_m[j]/self.Δφ**2
                Dj = self.cosφ[j]*self.cosφ_p[j]/self.Δφ**2
            B[j], D[j] = Bj, Dj
            C[j] = -(Bj + Dj)
        # Assemble banded matrix rows: (super, main, sub)
        self._ab = np.zeros((3, Nφ))
        self._ab[0, 1:] = D[:-1]
        self._ab[1, :]  = C
        self._ab[2, :-1]= B[1:]

    def solve(self, R: np.ndarray, south=0, north=0, west=0, east=0) -> np.ndarray:
        """
        Solve ∇²ψ = (1/(a² cosφ)) ∂/∂φ(cosφ ∂ψ/∂φ) + (1/(a² cos²φ)) ∂²ψ/∂λ² = R with Neumann BCs via DCT-II + tridiagonal φ solves.
        To avoid some extreme cases that cosφ is very close to 0, it actually solve following equation.
        --> cosφ∂/∂φ(cosφ ∂ψ/∂φ) + ∂²ψ/∂λ², and RHS = a² cos²φ * R.
        """
        # scale RHS
        RHS = self.a**2 * (self.cosφ**2)[:,None] * R
        # apply BCs
        if south is not None:
            RHS[0,:]  = 0.5*RHS[0,:]  + south * self.cosφ[0]*self.cosφ_p[0]/self.Δφ
        if north is not None:
            RHS[-1,:] = 0.5*RHS[-1,:] - north * self.cosφ[-1]*self.cosφ_m[-1]/self.Δφ
        if west is not None:
            RHS[:,0]  = 0.5*RHS[:,0]  + west/self.Δλ
        if east is not None:
            RHS[:,-1] = 0.5*RHS[:,-1] - east/self.Δλ
        # remove mean
        RHS -= RHS.mean()
        # DCT-II
        Rhat = dct(RHS, axis=1, type=2, norm='ortho')
        # solve per mode
        psihat = np.empty_like(Rhat)
        for m in range(self.Nλ):
            ab = self._ab.copy()
            ab[1,:] += self._lam_eig[m]
            if m == 0:
                # Build full matrix A from the banded representation "ab"
                A = np.diag(ab[1, :])
                A += np.diag(ab[0, 1:], k=1)
                A += np.diag(ab[2, :-1], k=-1)
                psihat[:,m] = lstsq(A, Rhat[:, m], cond=1e-8, overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')[0]
            else:
                psihat[:,m] = solve_banded((1,1), ab, Rhat[:,m], overwrite_ab=True, overwrite_b=True)
        # invert DCT
        psi = idct(psihat, axis=1, type=2, norm='ortho')
        return psi
    
    def _derivative(self, fλ: np.ndarray, fφ: np.ndarray):
        df_dλ, df_dφ = np.zeros_like(fλ), np.zeros_like(fφ)

        df_dλ[:,1:-1] = fλ[:,2:] - fλ[:,:-2]
        df_dλ[:,0]    = -3*fλ[:,0] + 4*fλ[:,1] - fλ[:,2]
        df_dλ[:,-1]   =  3*fλ[:,-1] - 4*fλ[:,-2] + fλ[:,-3]

        # φ-derivative
        df_dφ[1:-1,:] = fφ[2:,:] - fφ[:-2,:]
        df_dφ[0,:]    = -3*fφ[0,:] + 4*fφ[1,:] - fφ[2,:]
        df_dφ[-1,:]   =  3*fφ[-1,:] - 4*fφ[-2,:] + fφ[-3,:]

        return df_dλ / (2 * self.Δλ), df_dφ / (2 * self.Δφ)

    
    def compute_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        ucos = u * self.cosφ[:,None]
        dv_dλ, ducos_dφ = self._derivative(v, ucos)

        return (dv_dλ - ducos_dφ) / (self.a * self.cosφ)[:,None]
    
    def compute_divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        vcos = v * self.cosφ[:,None]
        du_dλ, dvcos_dφ = self._derivative(u, vcos)

        return (du_dλ + dvcos_dφ) / (self.a * self.cosφ)[:,None]
    
    def compute_uv_from_ψ(self, ψ):
        dψ_dλ, dψ_dφ = self._derivative(ψ, ψ)

        return -dψ_dφ / self.a, dψ_dλ / (self.a * self.cosφ)[:,None]
    
    def compute_uv_from_χ(self, χ):
        dχ_dλ, dχ_dφ = self._derivative(χ, χ)

        return dχ_dλ / (self.a * self.cosφ)[:,None], dχ_dφ / self.a


