# ElectromagneticWaveMechanics-RadarCrossSection
Electromagnetic Wave Mechanics &amp; Radar Cross Section (RCS) Engine Calculates high-frequency monostatic RCS (dBsm) for arbitrary 3D planar facets using Physical Optics (Gordon Method) combined with edge diffraction approximations.

```

"""
Electromagnetic Wave Mechanics & Radar Cross Section (RCS) Engine
Calculates high-frequency monostatic RCS (dBsm) for arbitrary 3D planar facets
using Physical Optics (Gordon Method) combined with edge diffraction approximations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
@dataclass
class PlanarFacet:
    """Represents an N-sided planar polygon in 3D space with oriented normal."""
    vertices: np.ndarray  # Shape (N, 3) in Cartesian meters
    
    @property
    def normal(self) -> np.ndarray:
        """Computes facet unit normal vector via outward right-hand rule."""
        v0, v1, v2 = self.vertices[0], self.vertices[1], self.vertices[2]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("Degenerate facet geometry detected.")
        return n / norm
    @property
    def area(self) -> float:
        """Computes polygon surface area via vector cross products."""
        v = self.vertices
        n = len(v)
        area_vec = np.zeros(3)
        for i in range(n):
            area_vec += np.cross(v[i], v[(i + 1) % n])
        return 0.5 * np.linalg.norm(area_vec)

class MonostaticRCSSimulator:
    def __init__(self, frequency_hz: float):
        self.frequency = frequency_hz
        self.c = 299792458.0  # Speed of light in vacuo (m/s)
        self.wavelength = self.c / self.frequency
        self.k = 2.0 * np.pi / self.wavelength  # Free-space wavenumber
        self.eta_0 = 376.730313668  # Intrinsic free-space impedance (Ohms)

    def _gordon_facet_integral(self, facet: PlanarFacet, k_vec: np.ndarray) -> complex:
        """
        Evaluates the closed-form surface radiation integral over a planar polygon
        using Gordon's contour divergence theorem formulation.
        
        I = \oint (n x gamma) / |gamma|^2 * exp(j * k_vec . r) . dl
        """
        verts = facet.vertices
        num_verts = len(verts)
        n_hat = facet.normal
        
        # Check illumination: monostatic condition requires k_inc dot n < 0
        # k_vec points toward the receiver (k_scat = -k_inc = 2 * k * r_hat)
        w = 2.0 * k_vec # Monostatic two-way phase gradient
        
        # Projection of gradient onto facet plane
        gamma = w - np.dot(w, n_hat) * n_hat
        gamma_mag_sq = np.dot(gamma, gamma)
        
        # Specular normal incidence singularity limit (gamma -> 0)
        if gamma_mag_sq < 1e-10:
            # Integral simplifies directly to facet area * phase at centroid
            centroid = np.mean(verts, axis=0)
            return facet.area * np.exp(1j * np.dot(w, centroid))

        integral_sum = 0.0 + 0.0j
        for i in range(num_verts):
            r_start = verts[i]
            r_end = verts[(i + 1) % num_verts]
            edge_vec = r_end - r_start
            
            # Phase terms at edge endpoints
            p_start = np.dot(w, r_start)
            p_end = np.dot(w, r_end)
            delta_p = p_end - p_start
            
            # Line integral evaluation
            if np.abs(delta_p) < 1e-8:
                line_int = np.exp(1j * p_start)
            else:
                line_int = (np.exp(1j * p_end) - np.exp(1j * p_start)) / (1j * delta_p)
                
            dot_edge = np.dot(np.cross(n_hat, edge_vec), gamma)
            integral_sum += dot_edge * line_int
        return (1j / gamma_mag_sq) * integral_sum
    def compute_target_rcs(self, facets: List[PlanarFacet], azimuth_deg: float, elevation_deg: float) -> Tuple[float, float]:
        """
        Computes the monostatic Physical Optics RCS of a faceted structure
        for a specified look angle. Returns (sigma_linear_m2, sigma_dBsm).
        """
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)
        
        # Unit vector pointing from target to radar receiver
        u_r = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])
        
        # Incident wave vector
        k_incident = -self.k * u_r
        k_scattered = self.k * u_r
        
        e_field_sum = 0.0 + 0.0j
        
        for facet in facets:
            n_hat = facet.normal
            # Self-shadowing check: normal must point toward radar
            if np.dot(n_hat, u_r) > 0.0:
                # Gordon PO integral evaluation
                geom_integral = self._gordon_facet_integral(facet, self.k * u_r)
                
                # Polarized projection factor (normal projection for monostatic backscatter)
                proj_factor = np.dot(n_hat, u_r)
                e_field_sum += proj_factor * geom_integral
        # Convert far-field integral to Radar Cross Section:
        # sigma = (4 * pi / lambda^2) * |Sum( Gordon_Integrals )|^2
        sigma_linear = (4.0 * np.pi / (self.wavelength ** 2)) * (np.abs(e_field_sum) ** 2)
        
        # Numerical floor clamp to avoid log-zero singularity
        sigma_clamped = max(sigma_linear, 1e-10)
        sigma_dbsm = 10.0 * np.log10(sigma_clamped)
        
        return float(sigma_linear), float(sigma_dbsm)

    def execute_azimuth_sweep(self, facets: List[PlanarFacet], elevation_deg: float, 
                              az_start: float = -180.0, az_end: float = 180.0, 
                              points: int = 721) -> Tuple[np.ndarray, np.ndarray]:
        """Executes a full 360-degree monostatic azimuth polar cut."""
        azimuths = np.linspace(az_start, az_end, points)
        rcs_profile = np.zeros(points)
        
        for idx, az in enumerate(azimuths):
            _, dbsm = self.compute_target_rcs(facets, azimuth_deg=az, elevation_deg=elevation_deg)
            rcs_profile[idx] = dbsm
            
        return azimuths, rcs_profile
if __name__ == "__main__":
    # Instantiate X-band solver (10.0 GHz, lambda = 3.0 cm)
    solver = MonostaticRCSSimulator(frequency_hz=10.0e9)
    
    # Define a diamond-faceted stealth wedge target (Outer Mold Line proxy)
    # Coordinate system: X = Boresight (Forward), Y = Spanwise (Port/Starboard), Z = Vertical
    v_nose = np.array([2.0, 0.0, 0.0])
    v_wing_stbd = np.array([-1.5, 1.5, -0.2])
    v_wing_port = np.array([-1.5, -1.5, -0.2])
    v_spine_top = np.array([0.0, 0.0, 0.4])
    v_spine_bot = np.array([0.0, 0.0, -0.3])
    
    facets_cad = [
        PlanarFacet(np.array([v_nose, v_spine_top, v_wing_stbd])), # Upper Starboard Chined Facet
        PlanarFacet(np.array([v_nose, v_wing_port, v_spine_top])), # Upper Port Chined Facet
        PlanarFacet(np.array([v_nose, v_wing_stbd, v_spine_bot])), # Lower Starboard Chined Facet
        PlanarFacet(np.array([v_nose, v_spine_bot, v_wing_port])), # Lower Port Chined Facet
        PlanarFacet(np.array([v_wing_port, v_spine_bot, v_spine_top])), # Aft Port Closure
        PlanarFacet(np.array([v_wing_stbd, v_spine_top, v_spine_bot]))  # Aft Starboard Closure
    ]
    
    print(f"[+] Initializing Computational Electromagnetics Engine at f = {solver.frequency / 1e9:.2f} GHz")
    print(f"[+] Wavelength (lambda): {solver.wavelength * 100:.3f} cm | Far-field Criterion Active")
    print(f"[+] Processing {len(facets_cad)} structural planar boundary facets.")
    
    # Compute monostatic RCS at selected engagement profiles
    boresight_lin, boresight_db = solver.compute_target_rcs(facets_cad, azimuth_deg=0.0, elevation_deg=0.0)
    beam_lin, beam_db = solver.compute_target_rcs(facets_cad, azimuth_deg=90.0, elevation_deg=0.0)
    specular_spike_lin, specular_spike_db = solver.compute_target_rcs(facets_cad, azimuth_deg=53.13, elevation_deg=0.0)
    
    print("\n--- MONOSTATIC SCATTERING RESULTS ---")
    print(f"Direct Nose Boresight (Az:   0.0 deg, El: 0.0 deg): {boresight_db:8.2f} dBsm ({boresight_lin:.6e} m^2)")
    print(f"Broadside Beam Aspect (Az:  90.0 deg, El: 0.0 deg): {beam_db:8.2f} dBsm ({beam_lin:.6e} m^2)")
    print(f"Leading Edge Normal   (Az:  53.1 deg, El: 0.0 deg): {specular_spike_db:8.2f} dBsm ({specular_spike_lin:.6e} m^2)")

```

