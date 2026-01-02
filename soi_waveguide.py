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
core_widths = np.linspace(0.4, 0.5, 11)  # 400 to 500 nm

# Wavelength
wl0 = 1.55  # microns
freq = 1.0 / wl0

# Computational domain
sy = 3.0  # 3 microns in y (vertical)
sz = 3.0  # 3 microns in z (horizontal)
resolution = 64  # pixels per micron

# Turn off verbose output
mp.verbosity(0)

def create_soi_geometry(width):
    """Create SOI waveguide geometry"""
    geometry = [
        # Start with air background
        mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, mp.inf), material=air),
        # SiO2 substrate (bottom half)
        mp.Block(center=mp.Vector3(0, -sy/4, 0), size=mp.Vector3(mp.inf, sy/2, mp.inf), material=sio2),
        # Silicon core on top of substrate
        mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(mp.inf, core_height, width), material=si)
    ]
    return geometry

def solve_te_mode(width, num_modes=4):
    
    geometry = create_soi_geometry(width)
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
    n_mode_guess = 0.5 * (n_si + n_sio2)  # Initial guess between core and cladding
    n_mode_min = n_sio2  # Must be above cladding index
    n_mode_max = n_si    # Must be below core index
    
    k_guess = freq * n_mode_guess
    k_min = freq * n_mode_min
    k_max = freq * n_mode_max
    
    # Find the fundamental TE mode (mode_num=1)
    k_mpb = ms.find_k(
        mp.NO_PARITY,
        freq,
        1,  # band_min
        1,  # band_max  
        mp.Vector3(1, 0, 0),  # k direction (propagation along x)
        tol,
        k_guess,
        k_min,
        k_max
    )
    
    neff = k_mpb[0] / freq
    
    # Get field components
    E = ms.get_efield(which_band=1)
    H = ms.get_hfield(which_band=1)
    P = ms.get_poynting(which_band=1)
    
    
    Ex = E[:, :, 0, 0]
    Ey = E[:, :, 0, 1]
    Ez = E[:, :, 0, 2]
    Px = 0.5 * np.real(P[:, :, 0, 0])
    
    # Get epsilon profile for visualization
    eps = ms.get_epsilon()
    
    return neff, Ex, Ey, Ez, Px, eps, ms

# Perform width sweep
print("="*70)
print("SOI WAVEGUIDE MODE SOLVER")
print("="*70)
print(f"Wavelength: {wl0*1000:.0f} nm")
print(f"Core: Si (n={n_si}), Substrate: SiO2 (n={n_sio2}), Cladding: Air (n={n_air})")
print(f"Core thickness: {core_height*1000:.0f} nm")
print("="*70)
print("\nSolving for fundamental TE mode at different widths...")
print("-"*70)

neff_results = []
field_data = []
widths_nm = core_widths * 1000

for i, width in enumerate(core_widths):
    print(f"Width: {width*1000:.1f} nm ", end="", flush=True)
    neff, Ex, Ey, Ez, Px, eps, ms = solve_te_mode(width)
    neff_results.append(neff)
    field_data.append((Ex, Ey, Ez, Px, eps))
    print(f"-> neff = {neff:.4f}")

print("-"*70)
print("Mode solving complete!")
print("="*70)


fig = plt.figure(figsize=(16, 11))

# Coordinate arrays
x_coords = np.linspace(-sy/2, sy/2, int(sy * resolution))
y_coords = np.linspace(-sz/2, sz/2, int(sz * resolution))

# Plot 1: Effective index vs width
ax1 = plt.subplot(3, 4, 1)
plt.plot(widths_nm, neff_results, 'b-o', linewidth=2.5, markersize=8)
plt.xlabel('Waveguide Width (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Effective Index (neff)', fontsize=11, fontweight='bold')
plt.title('Effective Index vs Width', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')

# Plot 2: Confinement factor estimate
ax2 = plt.subplot(3, 4, 2)
confinement = [(n - n_sio2) / (n_si - n_sio2) for n in neff_results]
plt.plot(widths_nm, confinement, 'r-s', linewidth=2.5, markersize=8)
plt.xlabel('Waveguide Width (nm)', fontsize=11, fontweight='bold')
plt.ylabel('Confinement Factor', fontsize=11, fontweight='bold')
plt.title('Mode Confinement', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')

# Plot 3: Summary info
ax3 = plt.subplot(3, 4, 3)
ax3.axis('off')
summary_text = f"""
SOI WAVEGUIDE RESULTS
{'='*35}

Parameters:
  λ = {wl0*1000:.0f} nm
  h = {core_height*1000:.0f} nm
  n_Si = {n_si}
  n_SiO2 = {n_sio2}
  n_Air = {n_air}

Width Sweep:
  Min: {widths_nm[0]:.0f} nm
  Max: {widths_nm[-1]:.0f} nm
  
neff Results:
  Min: {min(neff_results):.4f}
  Max: {max(neff_results):.4f}
  Δneff: {max(neff_results)-min(neff_results):.4f}

Resolution: {resolution} px/μm
"""
ax3.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Plot epsilon profile for middle width
ax4 = plt.subplot(3, 4, 4)
eps_mid = field_data[5][4]
im = plt.contourf(y_coords, x_coords, eps_mid, levels=50, cmap='viridis')
plt.xlabel('z (μm)', fontsize=10)
plt.ylabel('y (μm)', fontsize=10)
plt.title(f'Permittivity Profile\nw={widths_nm[5]:.0f} nm', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax4, label='ε')
plt.axis('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# Plot mode profiles for three selected widths
selected_indices = [0, 5, 10]  # First, middle, last

for plot_idx, width_idx in enumerate(selected_indices):
    Ex, Ey, Ez, Px, eps = field_data[width_idx]
    
    # Ey (dominant for TE)
    ax = plt.subplot(3, 4, 5 + plot_idx)
    im = plt.contourf(y_coords, x_coords, np.abs(Ey), levels=50, cmap='hot')
    plt.xlabel('z (μm)', fontsize=9)
    plt.ylabel('y (μm)', fontsize=9)
    plt.title(f'|Ey| - w={widths_nm[width_idx]:.0f}nm\nneff={neff_results[width_idx]:.4f}', 
              fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    
    # Ex component
    ax = plt.subplot(3, 4, 9 + plot_idx)
    im = plt.contourf(y_coords, x_coords, np.abs(Ex), levels=50, cmap='seismic')
    plt.xlabel('z (μm)', fontsize=9)
    plt.ylabel('y (μm)', fontsize=9)
    plt.title(f'|Ex| - w={widths_nm[width_idx]:.0f}nm', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

# Plot Poynting vector for middle width
ax = plt.subplot(3, 4, 8)
Px_mid = field_data[5][3]
im = plt.contourf(y_coords, x_coords, Px_mid, levels=50, cmap='RdBu_r')
plt.xlabel('z (μm)', fontsize=9)
plt.ylabel('y (μm)', fontsize=9)
plt.title(f'Poynting Vector (Px)\nw={widths_nm[5]:.0f}nm', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, label='Px')
plt.axis('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

plt.tight_layout()
plt.savefig('soi_waveguide_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved as 'soi_waveguide_analysis.png'")
plt.show()

# Print detailed results table
print("\n" + "="*70)
print("DETAILED RESULTS TABLE")
print("="*70)
print(f"{'Width (nm)':<15} {'neff':<15} {'Confinement':<20}")
print("-"*70)
for i, (w, n) in enumerate(zip(widths_nm, neff_results)):
    conf = (n - n_sio2) / (n_si - n_sio2)
    print(f"{w:<15.1f} {n:<15.4f} {conf:<20.3f}")
print("="*70)