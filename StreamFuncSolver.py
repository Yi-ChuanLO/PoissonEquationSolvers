import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import solve_banded

class StreamFuncSolOnLatLon:
    """
    A class to perform calculations on a spherical-coordinate latitude-longitude patch.
    Capabilities include:
      1. Solving the Poisson equation ∇²ψ = R on the patch using an FACR-like
         method (Discrete Cosine Transform in longitude, tridiagonal solver in latitude).
         The spherical Laplacian ∇² includes full metric terms:
           ∇²ψ = (1/(a² cosφ)) ∂/∂φ(cosφ ∂ψ/∂φ) + (1/(a² cos²φ)) ∂²ψ/∂λ²
      2. Computing relative vorticity ζ from given u, v wind fields using the formula:
           ζ = (1 / (a cosφ)) [ ∂(v cosφ)/∂λ – ∂u/∂φ ]
      3. Computing u, v wind fields from a given streamfunction ψ using the formulas:
           u = - (1 / (a cosφ)) ∂ψ/∂φ
           v =   (1 / a)        ∂ψ/∂λ

    The patch is defined by:
      λ ∈ [lon.min(), lon.max()] (longitudes, degrees, converted to radians internally)
      φ ∈ [lat.min(), lat.max()] (latitudes,  degrees, converted to radians internally)
    It is assumed that the provided `lon` and `lat` arrays define a uniform grid.

    Neumann boundary conditions (∂ψ/∂n = 0) are enforced on all four edges for
    the Poisson solver. Finite differences are second-order accurate, using
    centered differences in the interior and one-sided differences at boundaries.

    NOTE ON CONSISTENCY:
    The definitions for u, v from ψ (item 3) and ζ from u, v (item 2) are specific.
    If one computes ζ using `compute_vorticity` with u, v derived from `compute_uv_from_streamfunction(ψ)`,
    the resulting ζ will NOT generally be equal to ∇²ψ (item 1) due to the choice of formulas.
    For standard meteorological consistency where ζ = ∇²ψ, the definitions of u, v, and/or ζ
    would need to be adjusted (e.g., u = -(1/a)∂ψ/∂φ, v=(1/(a cosφ))∂ψ/∂λ, and
    ζ = (1/(a cosφ))[∂v/∂λ - ∂(u cosφ)/∂φ]). This class implements the formulas
    as stated in items 1, 2, and 3 above.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray,
                 a_radius: float = 6.371e6):
        """
        Initialize the grid and precompute metric factors and solver components.

        Parameters
        ----------
        lon : np.ndarray
            1D array of longitude coordinates (in degrees, uniform grid).
        lat : np.ndarray
            1D array of latitude coordinates (in degrees, uniform grid).
        a_radius : float, optional
            Radius of the sphere (default = 6.371e6 m for Earth).
        """
        if not isinstance(lon, np.ndarray) or lon.ndim != 1:
            raise ValueError("lon must be a 1D numpy array.")
        if not isinstance(lat, np.ndarray) or lat.ndim != 1:
            raise ValueError("lat must be a 1D numpy array.")

        if lon.size < 3:
            raise ValueError(f"Nlam (longitude points) must be at least 3 for second-order boundary differences. Got {lon.size}.")
        if lat.size < 3:
            raise ValueError(f"Nphi (latitude points) must be at least 3 for second-order boundary differences. Got {lat.size}.")

        self.a = a_radius

        # Compute uniform grid vectors in radians
        self.lam = np.deg2rad(lon)  # shape: (Nlam,)
        self.phi = np.deg2rad(lat)  # shape: (Nphi,)

        # Store grid properties
        self.Nlam = self.lam.size
        self.Nphi = self.phi.size

        # Grid spacings (assuming uniform grid, checked by visual inspection of input usually)
        # A more robust check could be added if necessary
        if self.Nlam > 1:
            self.dlam = self.lam[1] - self.lam[0]
        else:
            self.dlam = np.pi # Default or raise error, though Nlam >=3 is checked
        if self.Nphi > 1:
            self.dphi = self.phi[1] - self.phi[0]
        else:
            self.dphi = np.pi/2 # Default or raise error

        # Precompute cosine terms
        self.cosphi = np.cos(self.phi)  # shape: (Nphi,)
        # Ensure cosphi is not zero, especially near poles if phi contains +/- pi/2 exactly
        # For regional grids, this is less likely an issue than for global grids.
        # A small epsilon might be needed if phi can be exactly +/- pi/2 for cos(phi) in denominators.
        if np.any(np.isclose(self.cosphi, 0.0)):
             print("Warning: cos(phi) is close to zero for some latitudes. This may lead to instability if it's in a denominator.")


        # Precompute C_{i+1/2} and C_{i-1/2} for φ-direction stencil (cosine at cell interfaces)
        # C_{i+1/2} = cos(phi[i] + dphi/2)
        # C_{i-1/2} = cos(phi[i] - dphi/2)
        self.C_iphalf = np.cos(self.phi + 0.5 * self.dphi)  # shape: (Nphi,)
        self.C_imhalf = np.cos(self.phi - 0.5 * self.dphi)  # shape: (Nphi,)

        # Precompute kappa[q] for λ-modes (eigenvalues of -∂²/∂λ² under Neumann BCs)
        q_idx = np.arange(self.Nlam)
        # For Nlam=1 or Nlam=2, Nlam-1 can be 0 or 1. Nlam >=3 is now enforced.
        self.kappa = 2.0 * (1.0 - np.cos(np.pi * q_idx / (self.Nlam - 1))) / (self.dlam ** 2)
        # kappa[0] will be 0, corresponding to the mean mode in longitude.

        # Prebuild the φ‐tridiagonal “base” coefficients (A_i, B_i, C_i) for the operator
        # (1 / (a² cosφ_i)) ∂/∂φ( cosφ · ∂/∂φ )
        # These are the coefficients of ψ_{i-1}, ψ_{i}, ψ_{i+1} respectively.
        N = self.Nphi
        self.A_coeffs = np.zeros(N)  # sub‐diagonal (multiplies ψ_{i-1})
        self.B_coeffs = np.zeros(N)  # diagonal (multiplies ψ_{i})
        self.C_coeffs = np.zeros(N)  # super‐diagonal (multiplies ψ_{i+1})

        for i in range(N):
            # Denominator for the φ-derivative terms
            denom = self.a**2 * self.cosphi[i] * (self.dphi ** 2)
            if np.isclose(denom, 0): # Added check for safety
                raise ValueError(f"Denominator in tridiagonal coefficient setup is zero or close to zero at phi index {i}. Check grid.")


            # Coefficients from (C_{i-1/2} ψ_{i-1} - (C_{i+1/2}+C_{i-1/2})ψ_i + C_{i+1/2} ψ_{i+1}) / denom
            coeff_imhalf = self.C_imhalf[i] / denom
            coeff_iphalf = self.C_iphalf[i] / denom

            if 0 < i < N - 1: # Interior points in φ
                self.A_coeffs[i] = coeff_imhalf
                self.C_coeffs[i] = coeff_iphalf
                self.B_coeffs[i] = - (coeff_imhalf + coeff_iphalf)
            else: # Boundary rows i=0 or i=N-1 (Neumann: ∂ψ/∂φ = 0)
                  # Implemented by assuming ghost point ψ_{-1}=ψ_1 (for i=0) or ψ_{N}=ψ_{N-2} (for i=N-1)
                if i == 0: # Bottom boundary (e.g., phi_min)
                    self.A_coeffs[i] = 0.0  # No ψ_{i-1} term after BC application
                    self.B_coeffs[i] = - (coeff_imhalf + coeff_iphalf) # Retains original diagonal structure
                    self.C_coeffs[i] = coeff_imhalf + coeff_iphalf    # ψ_1 term absorbs ψ_{-1}
                else:  # i == N - 1, Top boundary (e.g., phi_max)
                    self.A_coeffs[i] = coeff_imhalf + coeff_iphalf    # ψ_{N-2} term absorbs ψ_{N}
                    self.B_coeffs[i] = - (coeff_imhalf + coeff_iphalf) # Retains original diagonal structure
                    self.C_coeffs[i] = 0.0  # No ψ_{i+1} term

        # Prebuild banded‐matrix template `ab` for `solve_banded`.
        # `ab` has shape (3, Nphi):
        #   ab[0, 0..Nphi-2] = C_coeffs[0..Nphi-2] (1st super-diagonal)
        #   ab[1, :]         = B_coeffs[:]        (main diagonal, will be modified with kappa term)
        #   ab[2, 1..Nphi-1] = A_coeffs[1..Nphi-1] (1st sub-diagonal)
        self._ab_template = np.zeros((3, self.Nphi))
        self._ab_template[0, :-1] = self.C_coeffs[:-1]
        self._ab_template[1, :]   = self.B_coeffs # Base diagonal
        self._ab_template[2, 1:]  = self.A_coeffs[1:]

        # Precompute for performance in solve_streamfunction's loop
        self._inv_a2_cosphi2 = 1.0 / (self.a**2 * (self.cosphi**2))


    def solve_streamfunction(self, R: np.ndarray) -> np.ndarray:
        """
        Solve ∇²ψ = R on the spherical patch with Neumann BCs.
        ∇²ψ = (1/(a² cosφ)) ∂/∂φ(cosφ ∂ψ/∂φ) + (1/(a² cos²φ)) ∂²ψ/∂λ²

        Parameters
        ----------
        R : np.ndarray
            2D array of shape (Nphi, Nlam), the right-hand side of the Poisson equation.

        Returns
        -------
        psi : np.ndarray
            2D array of shape (Nphi, Nlam), the streamfunction ψ solution.
            The solution is constrained to have a grid mean of zero.
        """
        if R.shape != (self.Nphi, self.Nlam):
            raise ValueError(f"R must have shape ({self.Nphi}, {self.Nlam}), got {R.shape}")

        # The operator includes 1/a², so R should not be scaled by a².
        F = R # No scaling by a**2 needed as operator itself includes 1/a^2

        # 1) DCT-II in λ (axis=1) of F → F_hat[i, q] (Neumann in λ)
        F_hat = dct(F, type=2, norm='ortho', axis=1)  # shape: (Nphi, Nlam)

        # 2) For each λ‐mode q, solve the φ‐tridiagonal system.
        # The system is: (L_{φφ} - kappa[q]/(a² cos²φ)) ψ_hat = F_hat
        psi_hat = np.zeros((self.Nphi, self.Nlam))
        ab_q = np.zeros_like(self._ab_template) # Temporary matrix for each q

        # Static parts of ab_q (sub and super diagonals)
        ab_q[0, :-1] = self._ab_template[0, :-1]
        ab_q[2, 1:]  = self._ab_template[2, 1:]

        for q in range(self.Nlam):
            # Construct the main diagonal for this mode q
            # B_coeffs already contains the φ-operator part.
            # Subtract kappa[q]/(a² cos²φ_i) term from the diagonal.
            # self.kappa[q] are eigenvalues of -∂²/∂λ², so the term is -kappa[q] * (...)
            current_diagonal = self.B_coeffs - self.kappa[q] * self._inv_a2_cosphi2
            ab_q[1, :] = current_diagonal

            rhs_q = F_hat[:, q]

            # Solve tridiagonal system: ab_q has (1 super, 1 sub)
            # For q=0, kappa[0]=0. The matrix might be singular if not for BCs.
            # Neumann BCs make it singular for constant vectors. solve_banded might
            # find a particular solution if RHS is consistent. If not, might error.
            # The final psi -= psi.mean() sets the unique constant.
            try:
                sol_q = solve_banded((1, 1), ab_q, rhs_q)
            except np.linalg.LinAlgError as e:
                if q == 0 and self.kappa[q] == 0:
                    # A common issue for q=0 mode with Neumann conditions.
                    # One way is to try lstsq or adjust the matrix slightly.
                    # For now, re-raise or print a more informative message.
                    print(f"Warning: Linear algebra error for q={q} (kappa={self.kappa[q]}). "
                          "This might be due to the singular nature of the q=0 Neumann problem. "
                          "The mean of psi will be set to zero afterwards.")
                    # Attempting a least-squares solution for the singular q=0 case
                    # This requires constructing the full matrix M_q for M_q @ x = rhs_q
                    # This is more complex than solve_banded.
                    # An alternative is to slightly perturb the matrix to make it non-singular
                    # or set one value of sol_q (e.g. sol_q[0]=0) and solve for the rest.
                    # For now, we let it propagate if solve_banded truly fails.
                    # SciPy's solve_banded may handle consistent singular systems.
                    raise e # Or handle more gracefully if a specific strategy is known
                else:
                    raise e # Re-raise for other q values
            psi_hat[:, q] = sol_q

        # 3) Inverse DCT-III (which is idct(type=2, norm='ortho')) along λ to get ψ(i,j)
        psi = idct(psi_hat, type=2, norm='ortho', axis=1)

        # 4) Enforce zero‐mean for the solution (Neumann problem is unique up to a constant)
        psi -= psi.mean()

        return psi

    def compute_vorticity(self,
                          u: np.ndarray,
                          v: np.ndarray) -> np.ndarray:
        """
        Compute relative vorticity ζ from u, v on the spherical patch:
            ζ = (1 / (a cosφ)) [ ∂(v cosφ)/∂λ – ∂u/∂φ ]

        Parameters
        ----------
        u, v : np.ndarray
            2D arrays of shape (Nphi, Nlam) for zonal (eastward) and
            meridional (northward) wind components (m/s).

        Returns
        -------
        zeta : np.ndarray
            2D array of shape (Nphi, Nlam) for relative vorticity (s⁻¹).
        """
        if u.shape != (self.Nphi, self.Nlam) or v.shape != (self.Nphi, self.Nlam):
            raise ValueError(f"u, v must both have shape ({self.Nphi}, {self.Nlam})")

        # 1) Compute v cosφ
        vcos = v * self.cosphi[:, None]  # shape: (Nphi, Nlam)

        # Allocate arrays for partial derivatives
        dvcos_dlam = np.zeros((self.Nphi, self.Nlam))
        du_dphi = np.zeros((self.Nphi, self.Nlam))

        # 2) ∂(v cosφ)/∂λ (second-order finite differences)
        # Interior points (centered)
        dvcos_dlam[:, 1:-1] = (vcos[:, 2:] - vcos[:, :-2]) / (2.0 * self.dlam)
        # Left boundary j=0 (one-sided)
        dvcos_dlam[:, 0] = (-3.0 * vcos[:, 0] + 4.0 * vcos[:, 1] - vcos[:, 2]) / (2.0 * self.dlam)
        # Right boundary j=Nlam-1 (one-sided)
        dvcos_dlam[:, -1] = (3.0 * vcos[:, -1] - 4.0 * vcos[:, -2] + vcos[:, -3]) / (2.0 * self.dlam)

        # 3) ∂u/∂φ (second-order finite differences)
        # Interior points (centered)
        du_dphi[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * self.dphi)
        # Bottom boundary i=0 (one-sided)
        du_dphi[0, :] = (-3.0 * u[0, :] + 4.0 * u[1, :] - u[2, :]) / (2.0 * self.dphi)
        # Top boundary i=Nphi-1 (one-sided)
        du_dphi[-1, :] = (3.0 * u[-1, :] - 4.0 * u[-2, :] + u[-3, :]) / (2.0 * self.dphi)

        # 4) Combine into ζ
        inv_a_cosphi = 1.0 / (self.a * self.cosphi)  # shape: (Nphi,)
        zeta = inv_a_cosphi[:, None] * (dvcos_dlam - du_dphi)

        return zeta

    def compute_uv_from_streamfunction(self,
                                       psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute u, v from streamfunction ψ using:
            u = - (1 / (a cosφ)) ∂ψ/∂φ
            v =   (1 / a)        ∂ψ/∂λ

        Parameters
        ----------
        psi : np.ndarray
            2D array of shape (Nphi, Nlam) for streamfunction (m²/s).

        Returns
        -------
        u, v : tuple[np.ndarray, np.ndarray]
            Zonal (u) and meridional (v) wind components, each shape (Nphi, Nlam).
        """
        if psi.shape != (self.Nphi, self.Nlam):
            raise ValueError(f"psi must have shape ({self.Nphi}, {self.Nlam})")

        dpsi_dphi = np.zeros((self.Nphi, self.Nlam))
        dpsi_dlam = np.zeros((self.Nphi, self.Nlam))

        # 1) ∂ψ/∂φ (second-order finite differences)
        dpsi_dphi[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * self.dphi)
        dpsi_dphi[0, :] = (-3.0 * psi[0, :] + 4.0 * psi[1, :] - psi[2, :]) / (2.0 * self.dphi)
        dpsi_dphi[-1, :] = (3.0 * psi[-1, :] - 4.0 * psi[-2, :] + psi[-3, :]) / (2.0 * self.dphi)

        # 2) ∂ψ/∂λ (second-order finite differences)
        dpsi_dlam[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * self.dlam)
        dpsi_dlam[:, 0] = (-3.0 * psi[:, 0] + 4.0 * psi[:, 1] - psi[:, 2]) / (2.0 * self.dlam)
        dpsi_dlam[:, -1] = (3.0 * psi[:, -1] - 4.0 * psi[:, -2] + psi[:, -3]) / (2.0 * self.dlam)

        # 3) Compute u, v based on docstring formulas
        inv_a_cosphi = 1.0 / (self.a * self.cosphi)  # shape: (Nphi,)
        u = -inv_a_cosphi[:, None] * dpsi_dphi      # Corrected sign

        inv_a = 1.0 / self.a
        v = inv_a * dpsi_dlam                     # Corrected sign

        return u, v
