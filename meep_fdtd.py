import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Physical parameters
n_si = 3.476     # Silicon at 1550 nm
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
L = 10.0          # Coupling length: 10 microns

# Wavelength
wl0 = 1.55       
fcen = 1/wl0     
df = 0.1*fcen    

# Computational cell
sx = L + 4        
sy = 3.0          
sz = 4.0          
pml_thickness = 1.0

cell_size = mp.Vector3(sx, sy, sz)

# PML boundaries
pml_layers = [mp.PML(thickness=pml_thickness)]

# Waveguide positions (centered vertically, separated horizontally)
z_offset = (w + g0) / 2

# Geometry - Two coupled strip waveguides
geometry = [
    # SiO2 substrate (bottom half)
    mp.Block(
        center=mp.Vector3(0, -sy/4, 0),
        size=mp.Vector3(mp.inf, sy/2, mp.inf),
        material=sio2
    ),
    # Waveguide 1 (bottom, where we launch the mode)
    mp.Block(
        center=mp.Vector3(0, 0, -z_offset),
        size=mp.Vector3(L, h, w),
        material=si
    ),
    # Waveguide 2 (top)
    mp.Block(
        center=mp.Vector3(0, 0, z_offset),
        size=mp.Vector3(L, h, w),
        material=si
    )
]

# Source position (left side, in waveguide 1)
source_x = -L/2 - 0.5

# Eigenmode source - launches fundamental TE mode into waveguide 1
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

# Monitor positions (at the outputs, right side)
monitor_x = L/2 + 0.5

# Flux monitors at outputs
flux_regions = [
    # Output of waveguide 1 (input waveguide)
    mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, -z_offset),
        size=mp.Vector3(0, sy/2, sz/2)
    ),
    # Output of waveguide 2 (coupled waveguide)
    mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, z_offset),
        size=mp.Vector3(0, sy/2, sz/2)
    )
]

# Resolution
resolution = 30  # pixels per micron

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
flux_wg1 = sim.add_flux(fcen, df, 5, flux_regions[0])
flux_wg2 = sim.add_flux(fcen, df, 5, flux_regions[1])

print("="*70)
print("MEEP FDTD DIRECTIONAL COUPLER SIMULATION")
print("="*70)
print(f"Geometry:")
print(f"  Waveguide width: {w*1000:.0f} nm")
print(f"  Waveguide height: {h*1000:.0f} nm")
print(f"  Gap: {g0*1000:.0f} nm")
print(f"  Coupling length: {L:.1f} μm")
print(f"  Wavelength: {wl0*1000:.0f} nm")
print(f"\nCell size: {sx:.1f} × {sy:.1f} × {sz:.1f} μm³")
print(f"Resolution: {resolution} pixels/μm")
print(f"PML thickness: {pml_thickness} μm")
print("="*70)

# Field snapshots for animation
field_snapshots = []
snapshot_times = []

def collect_fields(sim):
    """Collect field data for visualization"""
    # Get Ey component (dominant for TE mode) in x-z plane at y=0
    ey = sim.get_array(center=mp.Vector3(0, 0, 0), 
                       size=mp.Vector3(sx, 0, sz), 
                       component=mp.Ey)
    field_snapshots.append(np.copy(ey))
    snapshot_times.append(sim.meep_time())
    
# Run simulation with field collection
runtime = 100  # Time units (adjust based on propagation)
snapshot_interval = 2

print("\nRunning FDTD simulation...")


# Run with field snapshots
sim.run(mp.at_every(snapshot_interval, collect_fields), until=runtime)

print(f"Simulation complete, Collected {len(field_snapshots)} snapshots")
# Get flux values
flux1 = mp.get_fluxes(flux_wg1)[0]
flux2 = mp.get_fluxes(flux_wg2)[0]
total_flux = flux1 + flux2

print("\nOutput Power Analysis:")

print(f"Waveguide 1 (input) output flux:  {flux1:.6e}")
print(f"Waveguide 2 (coupled) output flux: {flux2:.6e}")
print(f"Total output flux: {total_flux:.6e}")
print(f"\nPower splitting:")
print(f"  Waveguide 1: {flux1/total_flux*100:.1f}%")
print(f"  Waveguide 2: {flux2/total_flux*100:.1f}%")
print(f"\nCoupling efficiency: {flux2/total_flux*100:.1f}%")


# Visualize final field distribution
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 10))

# Get coordinate arrays
x_coords = np.linspace(-sx/2, sx/2, field_snapshots[0].shape[0])
z_coords = np.linspace(-sz/2, sz/2, field_snapshots[0].shape[1])

# Plot 1: Initial field distribution
ax1 = plt.subplot(2, 3, 1)
snapshot_idx_init = min(5, len(field_snapshots)-1)
im1 = plt.pcolormesh(z_coords, x_coords, np.real(field_snapshots[snapshot_idx_init]), 
                     cmap='RdBu_r', shading='gouraud')
plt.axhline(y=-L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='Waveguides')
plt.axhline(y=L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel('z (μm)', fontsize=11)
plt.ylabel('x (μm)', fontsize=11)
plt.title(f'Field at t={snapshot_times[snapshot_idx_init]:.1f}\n(Early)', fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Re(Ey)')
plt.xlim([-sz/2, sz/2])

# Plot 2: Middle field distribution
ax2 = plt.subplot(2, 3, 2)
snapshot_idx_mid = len(field_snapshots)//2
im2 = plt.pcolormesh(z_coords, x_coords, np.real(field_snapshots[snapshot_idx_mid]), 
                     cmap='RdBu_r', shading='gouraud')
plt.axhline(y=-L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
plt.axhline(y=L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel('z (μm)', fontsize=11)
plt.ylabel('x (μm)', fontsize=11)
plt.title(f'Field at t={snapshot_times[snapshot_idx_mid]:.1f}\n(Middle)', fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Re(Ey)')
plt.xlim([-sz/2, sz/2])

# Plot 3: Final field distribution
ax3 = plt.subplot(2, 3, 3)
im3 = plt.pcolormesh(z_coords, x_coords, np.real(field_snapshots[-1]), 
                     cmap='RdBu_r', shading='gouraud')
plt.axhline(y=-L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
plt.axhline(y=L/2, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel('z (μm)', fontsize=11)
plt.ylabel('x (μm)', fontsize=11)
plt.title(f'Field at t={snapshot_times[-1]:.1f}\n(Final)', fontsize=11, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='Re(Ey)')
plt.xlim([-sz/2, sz/2])

# Plot 4: Field intensity along x at z=-z_offset (waveguide 1)
ax4 = plt.subplot(2, 3, 4)
z_idx_wg1 = np.argmin(np.abs(z_coords + z_offset))
intensity_wg1 = np.abs(field_snapshots[-1][:, z_idx_wg1])**2
plt.plot(x_coords, intensity_wg1, 'b-', linewidth=2)
plt.axvspan(-L/2, L/2, alpha=0.2, color='cyan', label='Coupling region')
plt.xlabel('x (μm)', fontsize=11)
plt.ylabel('Intensity |Ey|²', fontsize=11)
plt.title('Intensity in Waveguide 1 (Input)', fontsize=11, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Field intensity along x at z=+z_offset (waveguide 2)
ax5 = plt.subplot(2, 3, 5)
z_idx_wg2 = np.argmin(np.abs(z_coords - z_offset))
intensity_wg2 = np.abs(field_snapshots[-1][:, z_idx_wg2])**2
plt.plot(x_coords, intensity_wg2, 'r-', linewidth=2)
plt.axvspan(-L/2, L/2, alpha=0.2, color='cyan', label='Coupling region')
plt.xlabel('x (μm)', fontsize=11)
plt.ylabel('Intensity |Ey|²', fontsize=11)
plt.title('Intensity in Waveguide 2 (Coupled)', fontsize=11, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Power distribution summary
ax6 = plt.subplot(2, 3, 6)
bars = plt.bar(['WG1\n(Input)', 'WG2\n(Coupled)'], 
               [flux1/total_flux*100, flux2/total_flux*100],
               color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
plt.ylabel('Power (%)', fontsize=11, fontweight='bold')
plt.title('Output Power Distribution', fontsize=12, fontweight='bold')
plt.ylim([0, 100])
plt.grid(True, axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fdtd_coupler_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'fdtd_coupler_analysis.png'")
plt.show()

# Field prop animation 
print("\nCreating field propagation animation...")
fig_anim, ax_anim = plt.subplots(figsize=(12, 6))

def animate(i):
    ax_anim.clear()
    im = ax_anim.pcolormesh(z_coords, x_coords, np.real(field_snapshots[i]), 
                            cmap='RdBu_r', shading='gouraud', vmin=-0.5, vmax=0.5)
    ax_anim.axhline(y=-L/2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    ax_anim.axhline(y=L/2, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    
    # Waveguid Pos
    ax_anim.axhline(y=0, color='white', linestyle=':', linewidth=1, alpha=0.5)
    ax_anim.text(sz/2*0.9, -z_offset, 'WG1', color='white', fontsize=10, 
                ha='right', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.5))
    ax_anim.text(sz/2*0.9, z_offset, 'WG2', color='white', fontsize=10, 
                ha='right', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    ax_anim.set_xlabel('z (μm)', fontsize=12, fontweight='bold')
    ax_anim.set_ylabel('x (μm)', fontsize=12, fontweight='bold')
    ax_anim.set_title(f'Field Propagation (Ey) - Time: {snapshot_times[i]:.1f}', 
                     fontsize=13, fontweight='bold')
    ax_anim.set_xlim([-sz/2, sz/2])
    return [im]

anim = animation.FuncAnimation(fig_anim, animate, frames=len(field_snapshots), 
                              interval=100, blit=False, repeat=True)


anim.save('field_propagation.gif', writer='pillow', fps=10, dpi=100)
print("Animation saved as 'field_propagation.gif'")
plt.close(fig_anim)

print("Finished")