import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
n_si = 3.476      # Silicon at 1550 nm
n_sio2 = 1.444    # Silicon dioxide
n_air = 1.0       # Air

# Materials
si = mp.Medium(index=n_si)
sio2 = mp.Medium(index=n_sio2)
air = mp.Medium(index=n_air)

# Waveguide geometry (in microns)
w = 0.45          # Width: 450 nm
h = 0.22          # Height: 220 nm
g0 = 0.2          # Gap: 200 nm

# Wavelength
wl0 = 1.55        # 1550 nm
fcen = 1/wl0      # Center frequency
df = 0.1*fcen     # Frequency width

# Computational parameters
resolution = 20   # pixels per micron
pml_thickness = 1.0

# Turn off verbose output for cleaner results
mp.verbosity(0)

print("="*80)
print("COUPLING LENGTH VERIFICATION: FDTD vs THEORY")
print("="*80)

# Step 1: Calculate theoretical coupling length using MPB
print("\nStep 1: Calculating theoretical coupling length using MPB...")
print("-"*80)

sy_mpb = 3.0
sz_mpb = 4.0
resolution_mpb = 64

z_offset = (w + g0) / 2

geometry_mpb = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, mp.inf), material=air),
    mp.Block(center=mp.Vector3(0, -sy_mpb/4, 0), 
            size=mp.Vector3(mp.inf, sy_mpb/2, mp.inf), 
            material=sio2),
    mp.Block(center=mp.Vector3(0, 0, -z_offset), 
            size=mp.Vector3(mp.inf, h, w), 
            material=si),
    mp.Block(center=mp.Vector3(0, 0, z_offset), 
            size=mp.Vector3(mp.inf, h, w), 
            material=si)
]

geometry_lattice = mp.Lattice(size=mp.Vector3(0, sy_mpb, sz_mpb))

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry_mpb,
    resolution=resolution_mpb,
    num_bands=2,
)

ms.init_params(mp.NO_PARITY, True)

# Find even and odd modes
tol = 1e-6
n_mode_guess = 0.5 * (n_si + n_sio2)
n_mode_min = n_sio2
n_mode_max = n_si

k_guess = fcen * n_mode_guess
k_min = fcen * n_mode_min
k_max = fcen * n_mode_max

k_mpb = ms.find_k(mp.NO_PARITY, fcen, 1, 2, mp.Vector3(1, 0, 0), 
                  tol, k_guess, k_min, k_max)

neff_even = k_mpb[0] / fcen
neff_odd = k_mpb[1] / fcen
delta_neff = neff_even - neff_odd
Lc_theory = wl0 / (2 * delta_neff)

print(f"Even mode neff: {neff_even:.6f}")
print(f"Odd mode neff:  {neff_odd:.6f}")
print(f"Δneff:          {delta_neff:.6f}")
print(f"Theoretical Lc: {Lc_theory:.3f} μm")
print("-"*80)

# Step 2: FDTD simulations for different coupling lengths
print(f"\nStep 2: Running FDTD simulations for multiple coupling lengths...")
print("-"*80)

# Define coupling length sweep
L_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
L_values = [ratio * Lc_theory for ratio in L_ratios]

# Storage for results
through_powers = []
cross_powers = []
total_powers = []

def run_fdtd_simulation(L):
    """Run FDTD simulation for given coupling length"""
    
    # Cell dimensions
    sx = L + 4  # Extra space for source and PML
    sy = 3.0
    sz = 4.0
    
    cell_size = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(thickness=pml_thickness)]
    
    # Geometry - Two coupled waveguides
    geometry = [
        mp.Block(center=mp.Vector3(0, -sy/4, 0),
                size=mp.Vector3(mp.inf, sy/2, mp.inf),
                material=sio2),
        mp.Block(center=mp.Vector3(0, 0, -z_offset),
                size=mp.Vector3(L, h, w),
                material=si),
        mp.Block(center=mp.Vector3(0, 0, z_offset),
                size=mp.Vector3(L, h, w),
                material=si)
    ]
    
    # Source at input of waveguide 1 (through port)
    source_x = -L/2 - 0.5
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(source_x, 0, -z_offset),
            size=mp.Vector3(0, sy, sz),
            eig_band=1,
            eig_parity=mp.NO_PARITY,
            eig_match_freq=True
        )
    ]
    
    # Output monitors
    monitor_x = L/2 + 0.5
    
    # Through port (same waveguide as input)
    through_region = mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, -z_offset),
        size=mp.Vector3(0, 1.5, 1.5)
    )
    
    # Cross port (coupled waveguide)
    cross_region = mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, z_offset),
        size=mp.Vector3(0, 1.5, 1.5)
    )
    
    # Initialize simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=resolution,
        default_material=air
    )
    
    # Add flux monitors
    through_flux = sim.add_flux(fcen, df, 1, through_region)
    cross_flux = sim.add_flux(fcen, df, 1, cross_region)
    
    # Run simulation with fixed time
    # Conservative estimate: propagation time through structure
    runtime = 50  # Fixed time units
    sim.run(until=runtime)
    
    # Get flux values
    through_power = mp.get_fluxes(through_flux)[0]
    cross_power = mp.get_fluxes(cross_flux)[0]
    
    return through_power, cross_power

# Run simulations for each length
for i, (L, ratio) in enumerate(zip(L_values, L_ratios)):
    print(f"Simulating L = {ratio:.2f}Lc ({L:.3f} μm)...", end=" ", flush=True)
    
    through_power, cross_power = run_fdtd_simulation(L)
    total_power = through_power + cross_power
    
    through_powers.append(through_power)
    cross_powers.append(cross_power)
    total_powers.append(total_power)
    
    through_pct = (through_power / total_power) * 100
    cross_pct = (cross_power / total_power) * 100
    
    print(f"Through: {through_pct:.1f}%, Cross: {cross_pct:.1f}%")

print("-"*80)

# Step 3: Calculate theoretical power transfer
print("\nStep 3: Computing theoretical power transfer curves...")
print("-"*80)

L_theory_fine = np.linspace(0, 2*Lc_theory, 200)
# Theoretical power transfer: P_cross = sin²(π*L/(2*Lc))
cross_power_theory = np.sin(np.pi * L_theory_fine / (2 * Lc_theory))**2
through_power_theory = np.cos(np.pi * L_theory_fine / (2 * Lc_theory))**2

# Normalize FDTD results
through_powers_norm = np.array(through_powers) / np.array(total_powers)
cross_powers_norm = np.array(cross_powers) / np.array(total_powers)

# Find maximum cross power location
max_cross_idx = np.argmax(cross_powers_norm)
Lc_fdtd = L_values[max_cross_idx]
error_percent = abs(Lc_fdtd - Lc_theory) / Lc_theory * 100

print(f"Theoretical Lc:     {Lc_theory:.3f} μm")
print(f"FDTD maximum at:    {Lc_fdtd:.3f} μm")
print(f"Relative error:     {error_percent:.2f}%")
print("-"*80)

# Step 4: Visualization
print("\nStep 4: Generating comprehensive plots...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Power vs Length - Main comparison
ax1 = plt.subplot(2, 3, 1)
plt.plot(L_theory_fine, cross_power_theory*100, 'r-', linewidth=3, 
         label='Cross (Theory)', alpha=0.7)
plt.plot(L_theory_fine, through_power_theory*100, 'b-', linewidth=3, 
         label='Through (Theory)', alpha=0.7)
plt.plot(L_values, cross_powers_norm*100, 'ro', markersize=12, 
         markeredgecolor='darkred', markeredgewidth=2, label='Cross (FDTD)')
plt.plot(L_values, through_powers_norm*100, 'bs', markersize=12, 
         markeredgecolor='darkblue', markeredgewidth=2, label='Through (FDTD)')
plt.axvline(x=Lc_theory, color='green', linestyle='--', linewidth=2, 
            label=f'Theoretical Lc={Lc_theory:.2f}μm')
plt.axvline(x=Lc_fdtd, color='orange', linestyle=':', linewidth=2, 
            label=f'FDTD max at {Lc_fdtd:.2f}μm')
plt.xlabel('Coupling Length L (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Power (%)', fontsize=12, fontweight='bold')
plt.title('Power Transfer: FDTD vs Theory', fontsize=13, fontweight='bold')
plt.legend(fontsize=9, loc='best')
plt.grid(True, alpha=0.3)
plt.xlim([0, max(L_values)*1.1])
plt.ylim([0, 105])

# Plot 2: Cross port power - zoomed
ax2 = plt.subplot(2, 3, 2)
plt.plot(L_theory_fine, cross_power_theory*100, 'r-', linewidth=3, alpha=0.7)
plt.plot(L_values, cross_powers_norm*100, 'ro', markersize=12, 
         markeredgecolor='darkred', markeredgewidth=2)
plt.axvline(x=Lc_theory, color='green', linestyle='--', linewidth=2, alpha=0.7)
plt.axhline(y=100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
plt.xlabel('Coupling Length L (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Cross Port Power (%)', fontsize=12, fontweight='bold')
plt.title('Cross Port Power Detail', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim([0, max(L_values)*1.1])
plt.ylim([0, 105])

# Plot 3: Error analysis
ax3 = plt.subplot(2, 3, 3)
cross_error = (cross_powers_norm*100 - 
               np.sin(np.pi * np.array(L_values) / (2 * Lc_theory))**2 * 100)
through_error = (through_powers_norm*100 - 
                 np.cos(np.pi * np.array(L_values) / (2 * Lc_theory))**2 * 100)
plt.plot(L_values, cross_error, 'ro-', linewidth=2, markersize=10, 
         label='Cross port error')
plt.plot(L_values, through_error, 'bs-', linewidth=2, markersize=10, 
         label='Through port error')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Coupling Length L (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Error: FDTD - Theory (%)', fontsize=12, fontweight='bold')
plt.title('FDTD vs Theory Error', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 4: Summary info
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
summary_text = f"""
SIMULATION PARAMETERS
{'='*45}

Geometry:
  Width (w):        {w*1000:.0f} nm
  Height (h):       {h*1000:.0f} nm
  Gap (g₀):         {g0*1000:.0f} nm
  Wavelength (λ):   {wl0*1000:.0f} nm

Materials:
  n_Si:    {n_si}
  n_SiO₂:  {n_sio2}
  n_Air:   {n_air}

Mode Analysis:
  n_eff (even):     {neff_even:.6f}
  n_eff (odd):      {neff_odd:.6f}
  Δn_eff:           {delta_neff:.6f}

Coupling Length:
  Theoretical:      {Lc_theory:.3f} μm
  FDTD (max):       {Lc_fdtd:.3f} μm
  Error:            {error_percent:.2f}%

Resolution:         {resolution} px/μm
PML thickness:      {pml_thickness} μm
"""
ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Plot 5: Bar chart at specific lengths
ax5 = plt.subplot(2, 3, 5)
x_pos = np.arange(len(L_ratios))
width = 0.35
bars1 = plt.bar(x_pos - width/2, through_powers_norm*100, width, 
                label='Through', color='blue', alpha=0.7, edgecolor='black')
bars2 = plt.bar(x_pos + width/2, cross_powers_norm*100, width, 
                label='Cross', color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Coupling Length', fontsize=12, fontweight='bold')
plt.ylabel('Power (%)', fontsize=12, fontweight='bold')
plt.title('Power Distribution at Different Lengths', fontsize=13, fontweight='bold')
plt.xticks(x_pos, [f'{r:.2f}Lc' for r in L_ratios])
plt.legend(fontsize=10)
plt.grid(True, axis='y', alpha=0.3)
plt.ylim([0, 105])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

# Plot 6: Coupling efficiency vs normalized length
ax6 = plt.subplot(2, 3, 6)
L_normalized = np.array(L_values) / Lc_theory
L_norm_theory = L_theory_fine / Lc_theory
plt.plot(L_norm_theory, cross_power_theory*100, 'r-', linewidth=3, 
         label='Theory', alpha=0.7)
plt.plot(L_normalized, cross_powers_norm*100, 'ro', markersize=12, 
         markeredgecolor='darkred', markeredgewidth=2, label='FDTD')
plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, 
            label='L = Lc', alpha=0.7)
plt.xlabel('Normalized Length (L/Lc)', fontsize=12, fontweight='bold')
plt.ylabel('Cross Port Power (%)', fontsize=12, fontweight='bold')
plt.title('Coupling Efficiency vs L/Lc', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0, max(L_normalized)*1.1])
plt.ylim([0, 105])

plt.tight_layout()
plt.savefig('coupling_length_verification.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'coupling_length_verification.png'")
plt.show()

# Print detailed results table
print("\n" + "="*80)
print("DETAILED RESULTS TABLE")
print("="*80)
print(f"{'L (μm)':<10} {'L/Lc':<10} {'Through %':<12} {'Cross %':<12} {'Theory Cross %':<15}")
print("-"*80)
for i, (L, ratio) in enumerate(zip(L_values, L_ratios)):
    theory_cross = np.sin(np.pi * L / (2 * Lc_theory))**2 * 100
    print(f"{L:<10.3f} {ratio:<10.2f} {through_powers_norm[i]*100:<12.1f} "
          f"{cross_powers_norm[i]*100:<12.1f} {theory_cross:<15.1f}")
print("="*80)

# Analysis of discrepancies
print("\nANALYSIS OF DISCREPANCIES:")
print("-"*80)
print("Possible causes of FDTD vs Theory differences:")
print("  1. Finite resolution effects (30 pixels/μm)")
print("  2. Mode mismatch at source and boundaries")
print("  3. PML absorption affecting coupling region")
print("  4. 3D geometry vs 2D mode solver approximation")
print("  5. Numerical dispersion in FDTD")
print(f"\nMaximum error: {max(abs(cross_error)):.2f}%")
print(f"RMS error: {np.sqrt(np.mean(cross_error**2)):.2f}%")
print("="*80)