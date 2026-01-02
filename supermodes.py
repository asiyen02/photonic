import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep import mpb

# Physical parameters
n_si = 3.476      # Silicon at 1550 nm
n_sio2 = 1.444    # Silicon dioxide at 1550 nm
n_air = 1.0       # Air

# Materials
si = mp.Medium(index=n_si)
sio2 = mp.Medium(index=n_sio2)
air = mp.Medium(index=n_air)

# Waveguide geometry (in microns)
core_height = 0.22  # 220 nm
w = 0.45  # 450 nm nominal width

# Gap sweep
gaps = np.array([150, 200, 250, 300]) / 1000  # Convert nm to microns

# Wavelength
wl0 = 1.55  # microns
freq = 1.0 / wl0

# Computational domain
sy = 3.0  # 3 microns in y (vertical)
sz = 4.0  # 4 microns in z (horizontal) - wider for two waveguides
resolution = 64  # pixels per micron

# Turn off verbose output
mp.verbosity(0)

def create_coupled_waveguide_geometry(gap):
    """Create two coupled SOI waveguides with given gap"""
    # Position of waveguide centers
    z_offset = (w + gap) / 2
    
    geometry = [
        # Start with air background
        mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, mp.inf), material=air),
        # SiO2 substrate (bottom half)
        mp.Block(center=mp.Vector3(0, -sy/4, 0), 
                size=mp.Vector3(mp.inf, sy/2, mp.inf), 
                material=sio2),
        # Silicon waveguide 1 (left)
        mp.Block(center=mp.Vector3(0, 0, -z_offset), 
                size=mp.Vector3(mp.inf, core_height, w), 
                material=si),
        # Silicon waveguide 2 (right)
        mp.Block(center=mp.Vector3(0, 0, z_offset), 
                size=mp.Vector3(mp.inf, core_height, w), 
                material=si)
    ]
    return geometry

def solve_coupled_modes(gap, num_modes=2):
    """Solve for even and odd supermodes"""
    
    geometry = create_coupled_waveguide_geometry(gap)
    geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))
    
    # Initialize mode solver
    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=resolution,
        num_bands=num_modes,
    )
    
    ms.init_params(mp.NO_PARITY, True)
    
    # Parameters for find_k
    tol = 1e-6
    n_mode_guess = 0.5 * (n_si + n_sio2)
    n_mode_min = n_sio2
    n_mode_max = n_si
    
    k_guess = freq * n_mode_guess
    k_min = freq * n_mode_min
    k_max = freq * n_mode_max
    
    # Find both modes (even and odd)
    k_mpb = ms.find_k(
        mp.NO_PARITY,
        freq,
        1,  # band_min
        num_modes,  # band_max - get first two modes
        mp.Vector3(1, 0, 0),  # k direction (propagation along x)
        tol,
        k_guess,
        k_min,
        k_max
    )
    
    # Get effective indices
    neff_even = k_mpb[0] / freq  # Fundamental (even) mode
    neff_odd = k_mpb[1] / freq if len(k_mpb) > 1 else 0  # First excited (odd) mode
    
    # Get field profiles for both modes
    E_even = ms.get_efield(which_band=1)
    E_odd = ms.get_efield(which_band=2) if num_modes > 1 else None
    
    # Extract Ey component (dominant for TE)
    Ey_even = E_even[:, :, 0, 1]
    Ey_odd = E_odd[:, :, 0, 1] if E_odd is not None else None
    
    # Get epsilon profile
    eps = ms.get_epsilon()
    
    return neff_even, neff_odd, Ey_even, Ey_odd, eps, ms

# Perform gap sweep
print("="*70)
print("COUPLED WAVEGUIDE MODE SOLVER")
print("="*70)
print(f"Wavelength: {wl0*1000:.0f} nm")
print(f"Waveguide width: {w*1000:.0f} nm")
print(f"Core thickness: {core_height*1000:.0f} nm")
print(f"Core: Si (n={n_si}), Substrate: SiO2 (n={n_sio2}), Cladding: Air (n={n_air})")
print("="*70)
print("\nSolving for even and odd supermodes at different gaps...")
print("-"*70)

neff_even_results = []
neff_odd_results = []
coupling_lengths = []
field_data = []
gaps_nm = gaps * 1000

for i, gap in enumerate(gaps):
    print(f"Gap: {gap*1000:.0f} nm ", end="", flush=True)
    neff_even, neff_odd, Ey_even, Ey_odd, eps, ms = solve_coupled_modes(gap)
    
    # Calculate coupling length using supermode relation
    # Lc = λ / (2 * (neff_even - neff_odd))
    delta_neff = neff_even - neff_odd
    Lc = wl0 / (2 * delta_neff) if delta_neff > 0 else np.inf
    
    neff_even_results.append(neff_even)
    neff_odd_results.append(neff_odd)
    coupling_lengths.append(Lc)
    field_data.append((Ey_even, Ey_odd, eps))
    
    print(f"-> n_even = {neff_even:.4f}, n_odd = {neff_odd:.4f}, Lc = {Lc:.2f} μm")


print("Mode solving complete!")


# Create comprehensive plots
fig = plt.figure(figsize=(16, 10))

# Coordinate arrays
x_coords = np.linspace(-sy/2, sy/2, int(sy * resolution))
y_coords = np.linspace(-sz/2, sz/2, int(sz * resolution))

# Plot 1: Effective indices vs gap
ax1 = plt.subplot(2, 4, 1)
plt.plot(gaps_nm, neff_even_results, 'b-o', linewidth=2.5, markersize=8, label='Even mode')
plt.plot(gaps_nm, neff_odd_results, 'r-s', linewidth=2.5, markersize=8, label='Odd mode')
plt.xlabel('Gap (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Effective Index (neff)', fontsize=11, fontweight='bold')
plt.title('Effective Index vs Gap', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.4, linestyle='--')

# Plot 2: Coupling length vs gap
ax2 = plt.subplot(2, 4, 2)
plt.plot(gaps_nm, coupling_lengths, 'g-d', linewidth=2.5, markersize=8)
plt.xlabel('Gap (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Coupling Length Lc (μm)', fontsize=11, fontweight='bold')
plt.title('Coupling Length vs Gap', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.yscale('log')

# Plot 3: Delta neff vs gap
ax3 = plt.subplot(2, 4, 3)
delta_neff_values = np.array(neff_even_results) - np.array(neff_odd_results)
plt.plot(gaps_nm, delta_neff_values, 'm-^', linewidth=2.5, markersize=8)
plt.xlabel('Gap (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Δneff (n_even - n_odd)', fontsize=11, fontweight='bold')
plt.title('Index Difference vs Gap', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.yscale('log')

# Plot 4: Summary info
ax4 = plt.subplot(2, 4, 4)
ax4.axis('off')
summary_text = f"""
COUPLED WAVEGUIDE PARAMETERS
{'='*40}

Geometry:
  Width (w) = {w*1000:.0f} nm
  Height (h) = {core_height*1000:.0f} nm
  λ = {wl0*1000:.0f} nm
  
Materials:
  n_Si = {n_si}
  n_SiO2 = {n_sio2}
  n_Air = {n_air}

Gap Range:
  Min: {gaps_nm[0]:.0f} nm
  Max: {gaps_nm[-1]:.0f} nm

Coupling Length Range:
  Min: {min(coupling_lengths):.2f} μm
  Max: {max(coupling_lengths):.2f} μm

Resolution: {resolution} px/μm
"""
ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Plot mode profiles for first and last gaps only (to fit in 2x4 grid)
plot_gaps = [0, len(gaps)-1]  # First and last

for plot_idx, gap_idx in enumerate(plot_gaps):
    Ey_even, Ey_odd, eps = field_data[gap_idx]
    
    # Even mode (positions 5, 6)
    ax = plt.subplot(2, 4, 5 + plot_idx)
    im = plt.contourf(y_coords, x_coords, np.abs(Ey_even), levels=50, cmap='hot')
    
    # Overlay waveguide structure
    plt.contour(y_coords, x_coords, eps, levels=[n_sio2**2, n_si**2], 
                colors='cyan', linewidths=1.5, alpha=0.5)
    
    plt.xlabel('z (μm)', fontsize=9)
    plt.ylabel('y (μm)', fontsize=9)
    plt.title(f'Even Mode - Gap={gaps_nm[gap_idx]:.0f}nm\nn_eff={neff_even_results[gap_idx]:.4f}', 
              fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|Ey|')
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-1, 1])
    
    # Odd mode (positions 7, 8)
    ax = plt.subplot(2, 4, 7 + plot_idx)
    if Ey_odd is not None:
        im = plt.contourf(y_coords, x_coords, np.real(Ey_odd), levels=50, cmap='RdBu_r')
        
        # Overlay waveguide structure
        plt.contour(y_coords, x_coords, eps, levels=[n_sio2**2, n_si**2], 
                    colors='cyan', linewidths=1.5, alpha=0.5)
        
        plt.xlabel('z (μm)', fontsize=9)
        plt.ylabel('y (μm)', fontsize=9)
        plt.title(f'Odd Mode - Gap={gaps_nm[gap_idx]:.0f}nm\nn_eff={neff_odd_results[gap_idx]:.4f}', 
                  fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Re(Ey)')
        plt.axis('equal')
        plt.xlim([-2, 2])
        plt.ylim([-1, 1])

plt.tight_layout()
plt.savefig('coupled_waveguide_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved as 'coupled_waveguide_analysis.png'")
plt.show()

# Print detailed results table
print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)
print(f"{'Gap (nm)':<12} {'n_even':<12} {'n_odd':<12} {'Δneff':<15} {'Lc (μm)':<15}")
print("-"*80)
for i, gap in enumerate(gaps_nm):
    delta = neff_even_results[i] - neff_odd_results[i]
    print(f"{gap:<12.0f} {neff_even_results[i]:<12.4f} {neff_odd_results[i]:<12.4f} "
          f"{delta:<15.6f} {coupling_lengths[i]:<15.2f}")
print("="*80)

