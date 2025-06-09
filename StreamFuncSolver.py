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
        self.cos2 = self.cosφ ** 2
        self.sin2φ = np.sin(2*self.φ)
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
                Dj = self.cos2[j]/self.Δφ**2
            elif j == Nφ-1:
                Bj = self.cos2[j]/self.Δφ**2
                Dj = 0.0
            else:
                Bj = self.cos2[j]/self.Δφ**2 + self.sin2φ[j]/(4*self.Δφ)
                Dj = self.cos2[j]/self.Δφ**2 - self.sin2φ[j]/(4*self.Δφ)
            B[j], D[j] = Bj, Dj
            C[j] = -(Bj + Dj)
        # Assemble banded matrix rows: (super, main, sub)
        self._ab = np.zeros((3, Nφ))
        self._ab[0, 1:] = D[:-1]
        self._ab[1, :]  = C
        self._ab[2, :-1]= B[1:]

    def compute_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # ... identical to old implementation ...
        vcos = v * self.cosφ[:, None]
        dvcos_dλ = np.zeros_like(vcos)
        du_dφ = np.zeros_like(u)
        # λ-derivative
        dvcos_dλ[:,1:-1] = (vcos[:,2:] - vcos[:,:-2])/(2*self.Δλ)
        dvcos_dλ[:,0]    = (-3*vcos[:,0] +4*vcos[:,1] - vcos[:,2])/(2*self.Δλ)
        dvcos_dλ[:,-1]   = ( 3*vcos[:,-1] -4*vcos[:,-2] +vcos[:,-3])/(2*self.Δλ)
        # φ-derivative
        du_dφ[1:-1,:] = (u[2:,:] - u[:-2,:])/(2*self.Δφ)
        du_dφ[0,:]    = (-3*u[0,:] +4*u[1,:] - u[2,:])/(2*self.Δφ)
        du_dφ[-1,:]   = ( 3*u[-1,:]-4*u[-2,:]+ u[-3,:])/(2*self.Δφ)
        zeta = (dvcos_dλ - du_dφ) / (self.a * self.cosφ)[:,None]
        return zeta

    def solve(self, R: np.ndarray, south=0, north=0, west=0, east=0) -> np.ndarray:
        """
        Solve ∇²ψ = (1/(a² cosφ)) ∂/∂φ(cosφ ∂ψ/∂φ) + (1/(a² cos²φ)) ∂²ψ/∂λ² = R with Neumann BCs via DCT-II + tridiagonal φ solves.
        To avoid some extreme cases that cosφ is very close to 0, it actually solve following equation.
        --> cos²φ∂²ψ/∂φ² - 0.5*sin(2φ)∂ψ/∂φ + ∂²ψ/∂λ², and RHS = a² cos²φ * R.
        """
        # scale RHS
        RHS = self.a**2 * self.cos2[:,None] * R
        # apply BCs
        if south is not None:
            RHS[0,:]  = 0.5*RHS[0,:]  + south * (self.cos2[0]/self.Δφ + self.sin2φ[0]/4)
        if north is not None:
            RHS[-1,:] = 0.5*RHS[-1,:] - north * (self.cos2[-1]/self.Δφ - self.sin2φ[-1]/4)
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
            try:
                psihat[:,m] = solve_banded((1,1), ab, Rhat[:,m])
            except np.linalg.LinAlgError:
                # Singular matrix: use lstsq fallback
                A = np.zeros((self.Nφ, self.Nφ))
                # Fill tridiagonal matrix A from ab
                for i in range(self.Nφ):
                    A[i,i] = ab[1,i]
                    if i > 0:
                        A[i,i-1] = ab[2,i-1]
                    if i < self.Nφ-1:
                        A[i,i+1] = ab[0,i+1]
                psihat[:,m], *_ = lstsq(A, Rhat[:,m])

        # invert DCT
        psi = idct(psihat, axis=1, type=2, norm='ortho')
        return psi

    def compute_uv_from_streamfunction(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ... identical to old implementation ...
        dpsi_dφ = np.zeros_like(psi)
        dpsi_dλ = np.zeros_like(psi)
        dpsi_dφ[1:-1,:] = (psi[2:,:] - psi[:-2,:])/(2*self.Δφ)
        dpsi_dφ[0,:]    = (-3*psi[0,:]+4*psi[1,:]-psi[2,:])/(2*self.Δφ)
        dpsi_dφ[-1,:]   = ( 3*psi[-1,:]-4*psi[-2,:]+psi[-3,:])/(2*self.Δφ)
        dpsi_dλ[:,1:-1] = (psi[:,2:] - psi[:,:-2])/(2*self.Δλ)
        dpsi_dλ[:,0]    = (-3*psi[:,0]+4*psi[:,1]-psi[:,2])/(2*self.Δλ)
        dpsi_dλ[:,-1]   = ( 3*psi[:,-1]-4*psi[:,-2]+psi[:,-3])/(2*self.Δλ)
        u = -dpsi_dφ/(self.a * self.cosφ)[:,None]
        v =  dpsi_dλ/self.a
        return u, v
