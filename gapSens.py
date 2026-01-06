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
g0 = 0.2          # Nominal gap: 200 nm

# Wavelength
wl0 = 1.55        # 1550 nm
fcen = 1/wl0      # Center frequency
df = 0.1*fcen     # Frequency width

# Computational parameters
resolution = 25   # pixels per micron (reduced for speed)
pml_thickness = 1.0

# Turn off verbose output
mp.verbosity(0)

print("="*80)
print("GAP SENSITIVITY ANALYSIS - FABRICATION TOLERANCE STUDY")
print("="*80)

# Step 1: Find theoretical coupling length at nominal gap
print("\nStep 1: Calculating theoretical coupling length at nominal gap...")
print("-"*80)

sy_mpb = 3.0
sz_mpb = 4.0
resolution_mpb = 64

def calculate_coupling_length(gap):
    """Calculate theoretical coupling length for given gap"""
    z_offset = (w + gap) / 2
    
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
    Lc = wl0 / (2 * delta_neff)
    
    return Lc, neff_even, neff_odd, delta_neff

# Calculate nominal coupling length
Lc_nominal, neff_even_nom, neff_odd_nom, delta_neff_nom = calculate_coupling_length(g0)

print(f"Nominal gap (g₀):   {g0*1000:.0f} nm")
print(f"Even mode neff:     {neff_even_nom:.6f}")
print(f"Odd mode neff:      {neff_odd_nom:.6f}")
print(f"Δneff:              {delta_neff_nom:.6f}")
print(f"Coupling length:    {Lc_nominal:.3f} μm")

# Choose target: either full transfer (L=Lc) or 50/50 split (L=0.5*Lc)
TARGET = "full_transfer"  # Options: "full_transfer" or "50_50_split"

if TARGET == "full_transfer":
    L_fixed = Lc_nominal
    target_cross = 100  # Target 100% in cross port
    target_name = "Full Transfer (100% Cross)"
else:
    L_fixed = 0.5 * Lc_nominal
    target_cross = 50   # Target 50% in cross port
    target_name = "50/50 Split"

print(f"\nTarget configuration: {target_name}")
print(f"Fixed coupling length: {L_fixed:.3f} μm")
print("-"*80)

# Step 2: Gap sweep
print(f"\nStep 2: Gap sensitivity sweep around g₀ = {g0*1000:.0f} nm...")
print("-"*80)

# Define gap variations
gap_variations = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40]) / 1000  # nm to microns
gaps = g0 + gap_variations
gaps_nm = gaps * 1000

# Storage for results
through_powers = []
cross_powers = []
splitting_ratios = []
theoretical_Lc = []
theoretical_cross = []

def run_fdtd_simulation(gap, L):
    """Run FDTD simulation for given gap and coupling length"""
    
    z_offset = (w + gap) / 2
    
    # Cell dimensions
    sx = L + 4
    sy = 3.0
    sz = 4.0
    
    cell_size = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(thickness=pml_thickness)]
    
    # Geometry
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
    
    # Source at input of waveguide 1
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
    
    through_region = mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, -z_offset),
        size=mp.Vector3(0, 1.5, 1.5)
    )
    
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
    
    # Run simulation
    runtime = 50
    sim.run(until=runtime)
    
    # Get flux values
    through_power = mp.get_fluxes(through_flux)[0]
    cross_power = mp.get_fluxes(cross_flux)[0]
    
    return through_power, cross_power

# Run simulations for each gap
for i, gap in enumerate(gaps):
    gap_nm = gap * 1000
    delta_g = gap_nm - g0*1000
    
    print(f"Gap = {gap_nm:.0f} nm (g₀{delta_g:+.0f} nm)...", end=" ", flush=True)
    
    # Calculate theoretical parameters for this gap
    Lc_gap, _, _, _ = calculate_coupling_length(gap)
    theoretical_Lc.append(Lc_gap)
    
    # Theoretical cross power at fixed length L with this gap's Lc
    theory_cross = np.sin(np.pi * L_fixed / (2 * Lc_gap))**2 * 100
    theoretical_cross.append(theory_cross)
    
    # Run FDTD
    through_power, cross_power = run_fdtd_simulation(gap, L_fixed)
    total_power = through_power + cross_power
    
    through_pct = (through_power / total_power) * 100
    cross_pct = (cross_power / total_power) * 100
    splitting_ratio = cross_power / through_power if through_power > 0 else np.inf
    
    through_powers.append(through_pct)
    cross_powers.append(cross_pct)
    splitting_ratios.append(splitting_ratio)
    
    print(f"Cross: {cross_pct:.1f}%, Through: {through_pct:.1f}%")

print("-"*80)

# Convert to numpy arrays
through_powers = np.array(through_powers)
cross_powers = np.array(cross_powers)
theoretical_cross = np.array(theoretical_cross)

# Step 3: Determine tolerance window
print("\nStep 3: Analyzing fabrication tolerance...")
print("-"*80)

if TARGET == "full_transfer":
    # Good window: cross power > 90%
    tolerance_threshold = 90
    good_indices = np.where(cross_powers > tolerance_threshold)[0]
else:
    # Good window: cross power between 45% and 55%
    good_indices = np.where((cross_powers > 45) & (cross_powers < 55))[0]
    tolerance_threshold = None

if len(good_indices) > 0:
    gap_min_good = gaps_nm[good_indices[0]]
    gap_max_good = gaps_nm[good_indices[-1]]
    tolerance_range = gap_max_good - gap_min_good
    print(f"Good fabrication window:")
    print(f"  Gap range: {gap_min_good:.0f} nm to {gap_max_good:.0f} nm")
    print(f"  Tolerance: ±{tolerance_range/2:.0f} nm around optimal")
    print(f"  In good window: {len(good_indices)}/{len(gaps)} tested gaps")
else:
    print(f"Warning: No gaps met the tolerance criteria!")
    gap_min_good = None
    gap_max_good = None

print("-"*80)

# Step 4: Visualization
print("\nStep 4: Generating comprehensive plots...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Cross and Through power vs gap
ax1 = plt.subplot(2, 3, 1)
plt.plot(gaps_nm, cross_powers, 'ro-', linewidth=3, markersize=10, 
         label='Cross Port (FDTD)', markeredgecolor='darkred', markeredgewidth=2)
plt.plot(gaps_nm, through_powers, 'bs-', linewidth=3, markersize=10, 
         label='Through Port (FDTD)', markeredgecolor='darkblue', markeredgewidth=2)
plt.plot(gaps_nm, theoretical_cross, 'r--', linewidth=2, alpha=0.7, 
         label='Cross Port (Theory)')
plt.axvline(x=g0*1000, color='green', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Nominal g₀={g0*1000:.0f}nm')

if gap_min_good is not None:
    plt.axvspan(gap_min_good, gap_max_good, alpha=0.2, color='green', 
                label='Good window')

plt.xlabel('Gap (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Power (%)', fontsize=12, fontweight='bold')
plt.title(f'Power vs Gap\n(Fixed L = {L_fixed:.2f} μm)', fontsize=13, fontweight='bold')
plt.legend(fontsize=9, loc='best')
plt.grid(True, alpha=0.3)
plt.ylim([0, 105])

# Plot 2: Cross port power detail
ax2 = plt.subplot(2, 3, 2)
plt.plot(gaps_nm, cross_powers, 'ro-', linewidth=3, markersize=10,
         markeredgecolor='darkred', markeredgewidth=2)
plt.plot(gaps_nm, theoretical_cross, 'r--', linewidth=2, alpha=0.7)
plt.axhline(y=target_cross, color='orange', linestyle=':', linewidth=2, 
            label=f'Target: {target_cross}%')
plt.axvline(x=g0*1000, color='green', linestyle='--', linewidth=2, alpha=0.7)

if tolerance_threshold is not None:
    plt.axhline(y=tolerance_threshold, color='purple', linestyle=':', 
                linewidth=2, label=f'Threshold: {tolerance_threshold}%')

if gap_min_good is not None:
    plt.axvspan(gap_min_good, gap_max_good, alpha=0.2, color='green')

plt.xlabel('Gap (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Cross Port Power (%)', fontsize=12, fontweight='bold')
plt.title('Cross Port Detail', fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim([0, 105])

# Plot 3: Splitting ratio
ax3 = plt.subplot(2, 3, 3)
splitting_ratios_clipped = np.clip(splitting_ratios, 0, 20)  # Clip for visualization
plt.plot(gaps_nm, splitting_ratios_clipped, 'mo-', linewidth=3, markersize=10,
         markeredgecolor='purple', markeredgewidth=2)
plt.axvline(x=g0*1000, color='green', linestyle='--', linewidth=2, alpha=0.7)

if TARGET == "full_transfer":
    plt.axhline(y=10, color='orange', linestyle=':', linewidth=2, 
                label='Target: Cross >> Through')
else:
    plt.axhline(y=1, color='orange', linestyle=':', linewidth=2, 
                label='Target: 50/50 (ratio=1)')

if gap_min_good is not None:
    plt.axvspan(gap_min_good, gap_max_good, alpha=0.2, color='green')

plt.xlabel('Gap (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Splitting Ratio (Cross/Through)', fontsize=12, fontweight='bold')
plt.title('Splitting Ratio vs Gap', fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim([0, 20])

# Plot 4: Summary info
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

if gap_min_good is not None:
    tolerance_text = f"""
Good Window:
  {gap_min_good:.0f} - {gap_max_good:.0f} nm
  Tolerance: ±{tolerance_range/2:.0f} nm
  Coverage: {len(good_indices)}/{len(gaps)} gaps
"""
else:
    tolerance_text = "\nNo gaps in tolerance window!"

summary_text = f"""
FABRICATION TOLERANCE ANALYSIS
{'='*42}

Target Configuration:
  {target_name}
  Fixed length: {L_fixed:.3f} μm

Nominal Design:
  Gap (g₀):     {g0*1000:.0f} nm
  L_c:          {Lc_nominal:.3f} μm
  n_eff (even): {neff_even_nom:.4f}
  n_eff (odd):  {neff_odd_nom:.4f}

Gap Sweep:
  Range: {gaps_nm[0]:.0f} - {gaps_nm[-1]:.0f} nm
  Step:  {(gaps_nm[1]-gaps_nm[0]):.0f} nm
{tolerance_text}
Sensitivity:
  Δ(Cross%)/Δ(gap):
    ~{np.gradient(cross_powers, gaps_nm).mean():.2f} %/nm
"""
ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

# Plot 5: Theoretical coupling length vs gap
ax5 = plt.subplot(2, 3, 5)
plt.plot(gaps_nm, theoretical_Lc, 'go-', linewidth=3, markersize=10,
         markeredgecolor='darkgreen', markeredgewidth=2)
plt.axhline(y=L_fixed, color='orange', linestyle='--', linewidth=2, 
            label=f'Fixed L = {L_fixed:.2f} μm')
plt.axvline(x=g0*1000, color='green', linestyle='--', linewidth=2, alpha=0.7)

if gap_min_good is not None:
    plt.axvspan(gap_min_good, gap_max_good, alpha=0.2, color='green')

plt.xlabel('Gap (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Coupling Length Lc (μm)', fontsize=12, fontweight='bold')
plt.title('Theoretical Lc vs Gap', fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# Plot 6: Error between FDTD and theory
ax6 = plt.subplot(2, 3, 6)
error = cross_powers - theoretical_cross
plt.plot(gaps_nm, error, 'ko-', linewidth=2, markersize=8)
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.axvline(x=g0*1000, color='green', linestyle='--', linewidth=2, alpha=0.7)

if gap_min_good is not None:
    plt.axvspan(gap_min_good, gap_max_good, alpha=0.2, color='green')

plt.xlabel('Gap (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Error: FDTD - Theory (%)', fontsize=12, fontweight='bold')
plt.title('Cross Port Power Error', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gap_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'gap_sensitivity_analysis.png'")
plt.show()

# Print detailed results table
print("\n" + "="*80)
print("DETAILED RESULTS TABLE")
print("="*80)
print(f"{'Gap (nm)':<12} {'Δg (nm)':<12} {'Cross %':<12} {'Through %':<12} "
      f"{'Theory %':<12} {'In Window':<12}")
print("-"*80)
for i, gap_nm in enumerate(gaps_nm):
    delta_g = gap_nm - g0*1000
    in_window = "✓" if i in good_indices else "✗"
    print(f"{gap_nm:<12.0f} {delta_g:<12.0f} {cross_powers[i]:<12.1f} "
          f"{through_powers[i]:<12.1f} {theoretical_cross[i]:<12.1f} {in_window:<12}")
print("="*80)

print("\nFABRICATION RECOMMENDATIONS:")
print("-"*80)
if gap_min_good is not None:
    print(f"  • Target gap: {g0*1000:.0f} nm")
    print(f"  • Acceptable range: {gap_min_good:.0f} - {gap_max_good:.0f} nm")
    print(f"  • Fabrication tolerance: ±{tolerance_range/2:.0f} nm")
    print(f"  • This represents {tolerance_range/(g0*1000)*100:.1f}% relative tolerance")
else:
    print(f"  • WARNING: Design is very sensitive to gap variations!")
    print(f"  • Consider increasing coupling length or adjusting target")
print("="*80)
