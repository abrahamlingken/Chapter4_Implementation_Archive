#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FMC_3D_Solver_Fixed.py - Forward Monte Carlo Solver for 3D RTE
===============================================================

BUG FIXES APPLIED:
1. Track-Length Estimator: Energy deposited using exact voxel traversal
2. Data Race: Removed parallel=True, using single-threaded accumulation
3. Source Integral: Fixed calculus error (π/375 instead of π/150)

HIGH STATISTICS VERSION:
- 200 Million photons per case
- 100 batch progress tracking
- HighStats suffix for output files

Calculates the scalar incident radiation field G(x,y,z).

Author: Computational Physics Team
Date: 2024
"""

import numpy as np
from numba import njit
import os
import time
from scipy.stats import qmc

# =============================================================================
# CONFIGURATION
# =============================================================================

NX, NY, NZ = 50, 50, 50
DOMAIN = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
SOURCE_CENTER = np.array([0.5, 0.5, 0.5])
SOURCE_RADIUS = 0.2

# Russian Roulette settings
RR_THRESHOLD = 1e-4
RR_SURVIVAL_PROB = 0.1
RR_WEIGHT_BOOST = 10.0
RQMC_DIMENSIONS = 5  # source point (3) + launch direction (2)
RQMC_MAX_BATCH_POWER = 17  # 2**17 = 131072 photons per Sobol block

# HIGH STATISTICS: 200 Million photons per case
CASE_CONFIGS = {
    'B': {
        'name': 'Isotropic',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.0,
        'n_photons': 200_000_000  # UPDATED: 200 Million
    },
    'C': {
        'name': 'Anisotropic',
        'kappa': 0.5,
        'sigma_s': 4.5,
        'g': 0.8,
        'n_photons': 200_000_000  # UPDATED: 200 Million
    }
}


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def source_term(x, y, z):
    """Source term S(r) = max(0, 1 - 5*r)"""
    dx = x - SOURCE_CENTER[0]
    dy = y - SOURCE_CENTER[1]
    dz = z - SOURCE_CENTER[2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    s = 1.0 - 5.0 * r
    return s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True)
def sample_source_point():
    """Sample emission point within source sphere using rejection sampling."""
    while True:
        x = 0.3 + 0.4 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        z = 0.3 + 0.4 * np.random.random()
        
        dx = x - SOURCE_CENTER[0]
        dy = y - SOURCE_CENTER[1]
        dz = z - SOURCE_CENTER[2]
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if r < SOURCE_RADIUS:
            if np.random.random() < source_term(x, y, z):
                return x, y, z


@njit(cache=True, fastmath=True)
def sample_isotropic_direction():
    """Sample isotropic direction vector."""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    ux = sin_theta * np.cos(phi)
    uy = sin_theta * np.sin(phi)
    uz = cos_theta
    return ux, uy, uz


@njit(cache=True)
def seed_rng(seed):
    """Seed Numba's RNG for reproducible Monte Carlo runs."""
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def clamp_unit_interval(u):
    """Clamp quasi-random values away from 0 and 1 for stable transforms."""
    eps = 1e-15
    if u < eps:
        return eps
    if u > 1.0 - eps:
        return 1.0 - eps
    return u


@njit(cache=True, fastmath=True)
def invert_source_radius_cdf(u):
    """Invert F(t) = 4 t^3 - 3 t^4 on [0, 1], with r = 0.2 t."""
    u = clamp_unit_interval(u)
    lo = 0.0
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = 4.0 * mid * mid * mid - 3.0 * mid * mid * mid * mid
        if f_mid < u:
            lo = mid
        else:
            hi = mid
    return 0.2 * (0.5 * (lo + hi))


@njit(cache=True, fastmath=True)
def sample_isotropic_direction_from_uniforms(u1, u2):
    """Sample isotropic direction from two uniforms."""
    u1 = clamp_unit_interval(u1)
    u2 = clamp_unit_interval(u2)
    cos_theta = 2.0 * u1 - 1.0
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * u2
    ux = sin_theta * np.cos(phi)
    uy = sin_theta * np.sin(phi)
    uz = cos_theta
    return ux, uy, uz


@njit(cache=True, fastmath=True)
def sample_source_point_from_uniforms(u_radius, u_dir1, u_dir2):
    """Sample a source point exactly from the radial source density."""
    r = invert_source_radius_cdf(u_radius)
    dx, dy, dz = sample_isotropic_direction_from_uniforms(u_dir1, u_dir2)
    x = SOURCE_CENTER[0] + r * dx
    y = SOURCE_CENTER[1] + r * dy
    z = SOURCE_CENTER[2] + r * dz
    return x, y, z


@njit(cache=True, fastmath=True)
def sample_hg_direction(ux, uy, uz, g):
    """Sample new direction using Henyey-Greenstein phase function."""
    if abs(g) < 1e-10:
        return sample_isotropic_direction()
    
    if abs(g) > 0.999:
        cos_theta = 1.0 - 1e-6 * np.random.random()
    else:
        s = (1.0 - g*g) / (1.0 - g + 2.0 * g * np.random.random())
        cos_theta = (1.0 + g*g - s*s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))
    
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    phi = 2.0 * np.pi * np.random.random()
    
    # Build local coordinate system
    if abs(uz) < 0.999:
        perp_x = -uy
        perp_y = ux
        perp_z = 0.0
        norm = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        perp_x /= norm
        perp_y /= norm
    else:
        perp_x = 0.0
        perp_y = 1.0
        perp_z = 0.0
    
    perp2_x = uy * perp_z - uz * perp_y
    perp2_y = uz * perp_x - ux * perp_z
    perp2_z = ux * perp_y - uy * perp_x
    
    nx = sin_theta * (np.cos(phi) * perp_x + np.sin(phi) * perp2_x) + cos_theta * ux
    ny = sin_theta * (np.cos(phi) * perp_y + np.sin(phi) * perp2_y) + cos_theta * uy
    nz = sin_theta * (np.cos(phi) * perp_z + np.sin(phi) * perp2_z) + cos_theta * uz
    
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    return nx/norm, ny/norm, nz/norm


@njit(cache=True, fastmath=True)
def sample_hg_direction_from_uniforms(ux, uy, uz, g, u1, u2):
    """Sample HG-scattered direction from explicit uniforms."""
    u1 = clamp_unit_interval(u1)
    u2 = clamp_unit_interval(u2)
    if abs(g) < 1e-10:
        return sample_isotropic_direction_from_uniforms(u1, u2)

    if abs(g) > 0.999:
        cos_theta = 1.0 - 1e-6 * u1
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1)
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)
        cos_theta = max(-1.0, min(1.0, cos_theta))

    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * u2

    if abs(uz) < 0.999:
        perp_x = -uy
        perp_y = ux
        perp_z = 0.0
        norm = np.sqrt(perp_x * perp_x + perp_y * perp_y)
        perp_x /= norm
        perp_y /= norm
    else:
        perp_x = 0.0
        perp_y = 1.0
        perp_z = 0.0

    perp2_x = uy * perp_z - uz * perp_y
    perp2_y = uz * perp_x - ux * perp_z
    perp2_z = ux * perp_y - uy * perp_x

    nx = sin_theta * (np.cos(phi) * perp_x + np.sin(phi) * perp2_x) + cos_theta * ux
    ny = sin_theta * (np.cos(phi) * perp_y + np.sin(phi) * perp2_y) + cos_theta * uy
    nz = sin_theta * (np.cos(phi) * perp_z + np.sin(phi) * perp2_z) + cos_theta * uz

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / norm, ny / norm, nz / norm


@njit(cache=True, fastmath=True)
def distance_to_boundary(x, y, z, ux, uy, uz, xmin, xmax, ymin, ymax, zmin, zmax):
    """Calculate distance to domain boundary."""
    t_min = 1e15
    
    if ux > 1e-15:
        t = (xmax - x) / ux
        if t < t_min:
            t_min = t
    elif ux < -1e-15:
        t = (xmin - x) / ux
        if t < t_min:
            t_min = t
    
    if uy > 1e-15:
        t = (ymax - y) / uy
        if t < t_min:
            t_min = t
    elif uy < -1e-15:
        t = (ymin - y) / uy
        if t < t_min:
            t_min = t
    
    if uz > 1e-15:
        t = (zmax - z) / uz
        if t < t_min:
            t_min = t
    elif uz < -1e-15:
        t = (zmin - z) / uz
        if t < t_min:
            t_min = t
    
    return t_min


@njit(cache=True, fastmath=True)
def get_cell_indices(x, y, z, xmin, ymin, zmin, dx, dy, dz):
    """Get cell indices for current position."""
    ix = int((x - xmin) / dx)
    iy = int((y - ymin) / dy)
    iz = int((z - zmin) / dz)
    
    if ix < 0:
        ix = 0
    elif ix >= NX:
        ix = NX - 1
    
    if iy < 0:
        iy = 0
    elif iy >= NY:
        iy = NY - 1
    
    if iz < 0:
        iz = 0
    elif iz >= NZ:
        iz = NZ - 1
    
    return ix, iy, iz


@njit(cache=True, fastmath=True)
def accumulate_track_length_segment(
    G_field,
    x,
    y,
    z,
    ux,
    uy,
    uz,
    dist_travel,
    xmin,
    ymin,
    zmin,
    dx,
    dy,
    dz,
    cell_volume,
    weight,
):
    """Accumulate an exact track-length tally through crossed voxels."""
    tol = 1e-12
    huge = 1e30
    remaining = dist_travel

    ix, iy, iz = get_cell_indices(x, y, z, xmin, ymin, zmin, dx, dy, dz)

    if ux > tol:
        step_x = 1
        sx = (xmin + (ix + 1) * dx - x) / ux
        delta_x = dx / ux
    elif ux < -tol:
        step_x = -1
        sx = (xmin + ix * dx - x) / ux
        delta_x = -dx / ux
    else:
        step_x = 0
        sx = huge
        delta_x = huge

    if uy > tol:
        step_y = 1
        sy = (ymin + (iy + 1) * dy - y) / uy
        delta_y = dy / uy
    elif uy < -tol:
        step_y = -1
        sy = (ymin + iy * dy - y) / uy
        delta_y = -dy / uy
    else:
        step_y = 0
        sy = huge
        delta_y = huge

    if uz > tol:
        step_z = 1
        sz = (zmin + (iz + 1) * dz - z) / uz
        delta_z = dz / uz
    elif uz < -tol:
        step_z = -1
        sz = (zmin + iz * dz - z) / uz
        delta_z = -dz / uz
    else:
        step_z = 0
        sz = huge
        delta_z = huge

    if sx < 0.0:
        sx = 0.0
    if sy < 0.0:
        sy = 0.0
    if sz < 0.0:
        sz = 0.0

    while remaining > tol:
        ds = remaining
        if sx < ds:
            ds = sx
        if sy < ds:
            ds = sy
        if sz < ds:
            ds = sz

        if ds < 0.0:
            ds = 0.0

        G_field[ix, iy, iz] += weight * ds / cell_volume
        remaining -= ds

        if remaining <= tol:
            break

        sx -= ds
        sy -= ds
        sz -= ds

        if sx <= tol:
            ix += step_x
            if ix < 0 or ix >= NX:
                break
            sx += delta_x
        if sy <= tol:
            iy += step_y
            if iy < 0 or iy >= NY:
                break
            sy += delta_y
        if sz <= tol:
            iz += step_z
            if iz < 0 or iz >= NZ:
                break
            sz += delta_z


@njit(cache=True, fastmath=True)
def trace_photon(G_field, beta, albedo, g, xmin, xmax, ymin, ymax, zmin, zmax, 
                 dx, dy, dz, cell_volume, initial_weight):
    """
    Trace a single photon with an exact voxel track-length estimator.
    """
    x, y, z = sample_source_point()
    ux, uy, uz = sample_isotropic_direction()
    weight = initial_weight
    
    collision_count = 0
    max_collisions = 1000
    
    while collision_count < max_collisions and weight > 1e-10:
        dist_boundary = distance_to_boundary(x, y, z, ux, uy, uz, 
                                            xmin, xmax, ymin, ymax, zmin, zmax)
        
        xi = np.random.random()
        if xi < 1e-15:
            xi = 1e-15
        dist_collision = -np.log(1.0 - xi) / beta
        
        dist_travel = dist_collision if dist_collision < dist_boundary else dist_boundary

        accumulate_track_length_segment(
            G_field,
            x,
            y,
            z,
            ux,
            uy,
            uz,
            dist_travel,
            xmin,
            ymin,
            zmin,
            dx,
            dy,
            dz,
            cell_volume,
            weight,
        )
        
        if dist_collision >= dist_boundary:
            break

        x += ux * dist_travel
        y += uy * dist_travel
        z += uz * dist_travel
        
        # Survival biasing
        weight *= albedo
        
        # Russian Roulette
        if weight < RR_THRESHOLD:
            if np.random.random() < RR_SURVIVAL_PROB:
                weight *= RR_WEIGHT_BOOST
            else:
                break
        
        ux, uy, uz = sample_hg_direction(ux, uy, uz, g)
        collision_count += 1
    
    return collision_count


@njit(cache=True, fastmath=True)
def trace_photon_rqmc(
    G_field,
    beta,
    albedo,
    g,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    dx,
    dy,
    dz,
    cell_volume,
    initial_weight,
    uniforms,
):
    """Trace a photon using RQMC uniforms for the source and launch only."""
    x, y, z = sample_source_point_from_uniforms(uniforms[0], uniforms[1], uniforms[2])
    ux, uy, uz = sample_isotropic_direction_from_uniforms(uniforms[3], uniforms[4])
    weight = initial_weight

    collision_count = 0
    max_collisions = 1000

    while collision_count < max_collisions and weight > 1e-10:
        dist_boundary = distance_to_boundary(
            x, y, z, ux, uy, uz, xmin, xmax, ymin, ymax, zmin, zmax
        )

        xi = clamp_unit_interval(np.random.random())
        rr_u = clamp_unit_interval(np.random.random())
        dir_u1 = clamp_unit_interval(np.random.random())
        dir_u2 = clamp_unit_interval(np.random.random())

        dist_collision = -np.log(1.0 - xi) / beta
        dist_travel = dist_collision if dist_collision < dist_boundary else dist_boundary

        accumulate_track_length_segment(
            G_field,
            x,
            y,
            z,
            ux,
            uy,
            uz,
            dist_travel,
            xmin,
            ymin,
            zmin,
            dx,
            dy,
            dz,
            cell_volume,
            weight,
        )

        if dist_collision >= dist_boundary:
            break

        x += ux * dist_travel
        y += uy * dist_travel
        z += uz * dist_travel

        weight *= albedo

        if weight < RR_THRESHOLD:
            if rr_u < RR_SURVIVAL_PROB:
                weight *= RR_WEIGHT_BOOST
            else:
                break

        ux, uy, uz = sample_hg_direction_from_uniforms(ux, uy, uz, g, dir_u1, dir_u2)
        collision_count += 1

    return collision_count


# BUG FIX 2: Removed parallel=True and prange
@njit(cache=True, fastmath=False)
def run_monte_carlo(G_field, n_photons, beta, albedo, g, 
                   xmin, xmax, ymin, ymax, zmin, zmax,
                   dx, dy, dz, cell_volume, initial_weight):
    """
    Single-threaded accumulation to prevent data race.
    """
    for i in range(n_photons):  # Using range instead of prange
        trace_photon(G_field, beta, albedo, g, 
                    xmin, xmax, ymin, ymax, zmin, zmax,
                    dx, dy, dz, cell_volume, initial_weight)
    
    return n_photons


@njit(cache=True, fastmath=False)
def run_monte_carlo_rqmc(
    G_field,
    uniforms,
    beta,
    albedo,
    g,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    dx,
    dy,
    dz,
    cell_volume,
    initial_weight,
):
    """Run Monte Carlo using one quasi-random row per photon."""
    n_photons = uniforms.shape[0]
    for i in range(n_photons):
        trace_photon_rqmc(
            G_field,
            beta,
            albedo,
            g,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            dx,
            dy,
            dz,
            cell_volume,
            initial_weight,
            uniforms[i],
        )
    return n_photons


def iter_rqmc_block_sizes(n_photons, max_power=RQMC_MAX_BATCH_POWER):
    """Yield power-of-two Sobol block sizes summing to n_photons."""
    remaining = int(n_photons)
    while remaining > 0:
        block = 1 << min(max_power, int(np.floor(np.log2(remaining))))
        yield block
        remaining -= block


def generate_rqmc_block(n_block, seed):
    """Generate one scrambled Sobol block for early-collision sampling."""
    m = int(np.log2(n_block))
    sampler = qmc.Sobol(d=RQMC_DIMENSIONS, scramble=True, seed=seed)
    block = sampler.random_base2(m=m)
    return np.asarray(block, dtype=np.float64)


def save_aggregate_benchmark(
    case_key,
    mean_field,
    std_field,
    output_dir,
    output_suffix,
    metadata,
):
    """Save an aggregated benchmark field and its uncertainty summary."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}.npy')
    std_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}_std.npy')
    meta_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}_meta.npz')
    np.save(output_file, mean_field)
    np.save(std_file, std_field)
    np.savez(meta_file, **metadata)
    print(f"\nSaved aggregated mean field to: {output_file}")
    print(f"Saved aggregated std field to:  {std_file}")
    print(f"Saved aggregated metadata to:   {meta_file}")
    return output_file, std_file, meta_file


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_case(
    case_key,
    output_dir='MC3D_Results',
    n_photons_override=None,
    seed=None,
    output_suffix='_HighStats',
    sampler='mc',
):
    """Solve a single case using Forward Monte Carlo."""
    if case_key not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case_key}")
    
    config = CASE_CONFIGS[case_key]
    
    print("="*70)
    print(f"FMC 3D Solver (FIXED) - Case {case_key} ({config['name']})")
    print("HIGH STATISTICS VERSION - 200 Million Photons")
    print("="*70)

    sampler = str(sampler).lower()
    if sampler not in {'mc', 'rqmc'}:
        raise ValueError(f"Unknown sampler: {sampler}")
    
    kappa = config['kappa']
    sigma_s = config['sigma_s']
    g = config['g']
    n_photons = (
        int(n_photons_override)
        if n_photons_override is not None
        else config['n_photons']
    )
    
    beta = kappa + sigma_s
    albedo = sigma_s / beta if beta > 0 else 0.0
    
    print(f"Parameters:")
    print(f"  κ (absorption):    {kappa:.4f}")
    print(f"  σs (scattering):   {sigma_s:.4f}")
    print(f"  β (extinction):    {beta:.4f}")
    print(f"  ω (albedo):        {albedo:.4f}")
    print(f"  g (HG factor):     {g:.4f}")
    print(f"  Photons:           {n_photons:,}")
    print(f"  Grid:              {NX}x{NY}x{NZ}")
    print(f"  Sampler:           {sampler.upper()}")
    
    xmin, xmax = DOMAIN[0], DOMAIN[1]
    ymin, ymax = DOMAIN[2], DOMAIN[3]
    zmin, zmax = DOMAIN[4], DOMAIN[5]
    
    dx = (xmax - xmin) / NX
    dy = (ymax - ymin) / NY
    dz = (zmax - zmin) / NZ
    cell_volume = dx * dy * dz

    if seed is not None:
        seed_rng(int(seed))
    
    print(f"  Cell volume:       {cell_volume:.6e}")
    
    # BUG FIX 3: Correct source integral
    # Q_volume = ∫_0^0.2 (1-5r) * 4πr² dr = π/375
    Q_volume = np.pi / 375.0  # FIXED: was np.pi / 150.0
    
    # Total power = 4π * Q_volume (emission into 4π steradians)
    total_power = 4.0 * np.pi * Q_volume  # FIXED: proper 4π factor
    
    print(f"  Source integral:   {Q_volume:.6e} (volume, π/375)")
    print(f"  Total power:       {total_power:.6e} (4π emission)")
    
    initial_weight = total_power / n_photons
    print(f"  Initial weight:    {initial_weight:.6e}")
    
    G_field = np.zeros((NX, NY, NZ), dtype=np.float64)
    
    print("\nRunning Monte Carlo simulation...")
    print(f"Progress updates every 1% ({max(1, n_photons // 100):,} photons)")
    start_time = time.time()
    
    if sampler == 'rqmc':
        batch_sizes = list(iter_rqmc_block_sizes(n_photons))
    else:
        n_batches = 100
        batch_size = n_photons // n_batches
        batch_sizes = [batch_size] * n_batches
        remainder = n_photons - batch_size * n_batches
        if remainder > 0:
            batch_sizes[-1] += remainder

    n_batches = len(batch_sizes)

    for batch, current_batch_size in enumerate(batch_sizes):
        batch_start = time.time()
        G_batch = np.zeros((NX, NY, NZ), dtype=np.float64)

        if sampler == 'rqmc':
            block_seed = None if seed is None else int(seed) + batch
            uniforms = generate_rqmc_block(
                current_batch_size,
                seed=block_seed,
            )
            run_monte_carlo_rqmc(
                G_batch,
                uniforms,
                beta,
                albedo,
                g,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                dx,
                dy,
                dz,
                cell_volume,
                initial_weight,
            )
        else:
            run_monte_carlo(
                G_batch, current_batch_size, beta, albedo, g,
                xmin, xmax, ymin, ymax, zmin, zmax,
                dx, dy, dz, cell_volume, initial_weight
            )
        
        G_field += G_batch
        
        batch_time = time.time() - batch_start
        progress = (batch + 1) / n_batches * 100
        elapsed = time.time() - start_time
        eta = elapsed / (batch + 1) * (n_batches - batch - 1)
        
        print(f"  Batch {batch+1:3d}/{n_batches} ({progress:5.1f}%) | "
              f"{batch_time:6.2f}s | ETA: {eta/60:5.1f}m")
    
    total_elapsed = time.time() - start_time
    print(f"\nCompleted in {total_elapsed:.2f}s ({n_photons/total_elapsed:,.0f} photons/s)")
    
    ix_center = NX // 2
    iy_center = NY // 2
    iz_center = NZ // 2
    
    print(f"\nResults:")
    print(f"  G_center:          {G_field[ix_center, iy_center, iz_center]:.6f}")
    print(f"  G_max:             {G_field.max():.6f}")
    print(f"  G_min:             {G_field.min():.6e}")
    print(f"  G_mean:            {G_field.mean():.6e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}.npy')
    np.save(output_file, G_field)
    print(f"\nSaved to: {output_file}")
    
    metadata = {
        'case': case_key,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'beta': beta,
        'g': g,
        'n_photons': n_photons,
        'seed': -1 if seed is None else int(seed),
        'sampler': sampler,
        'Q_volume': float(Q_volume),
        'total_power': float(total_power),
        'raw_benchmark': True,
        'postprocessed': False,
        'tally_method': 'exact_voxel_traversal',
        'rqmc_dimensions': RQMC_DIMENSIONS if sampler == 'rqmc' else -1,
        'rqmc_mode': 'launch_only' if sampler == 'rqmc' else 'none',
        'G_center': float(G_field[ix_center, iy_center, iz_center]),
        'G_max': float(G_field.max()),
        'G_min': float(G_field.min())
    }
    
    meta_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}_meta.npz')
    np.savez(meta_file, **metadata)
    print(f"Metadata saved to: {meta_file}")
    
    return G_field


def solve_case_rqmc_replicates(
    case_key,
    output_dir='MC3D_Results',
    n_photons_override=None,
    seed=None,
    output_suffix='_RQMC_Aggregated',
    n_repeats=8,
):
    """Run multiple scrambled RQMC replicates and save their mean/std benchmark."""
    n_repeats = int(n_repeats)
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2 for aggregated RQMC benchmarks.")

    print("=" * 70)
    print(f"Aggregated RQMC Benchmark - Case {case_key}")
    print("=" * 70)
    print(f"  Repeats:           {n_repeats}")
    print(f"  Output suffix:     {output_suffix}")

    replicate_fields = []
    replicate_seeds = []
    repeat_photons = (
        int(n_photons_override)
        if n_photons_override is not None
        else int(CASE_CONFIGS[case_key]['n_photons'])
    )

    for rep_idx in range(n_repeats):
        rep_seed = None if seed is None else int(seed) + 1009 * rep_idx
        replicate_seeds.append(-1 if rep_seed is None else rep_seed)
        rep_suffix = f"{output_suffix}_scr{rep_idx + 1:02d}"
        field = solve_case(
            case_key,
            output_dir=output_dir,
            n_photons_override=repeat_photons,
            seed=rep_seed,
            output_suffix=rep_suffix,
            sampler='rqmc',
        )
        replicate_fields.append(field)
        print(f"Completed scramble {rep_idx + 1}/{n_repeats}\n")

    stack = np.stack(replicate_fields, axis=0)
    mean_field = np.mean(stack, axis=0)
    std_field = np.std(stack, axis=0, ddof=1)

    cfg = CASE_CONFIGS[case_key]
    kappa = cfg['kappa']
    sigma_s = cfg['sigma_s']
    g = cfg['g']
    beta = kappa + sigma_s
    Q_volume = np.pi / 375.0
    total_power = 4.0 * np.pi * Q_volume
    ix_center = NX // 2
    iy_center = NY // 2
    iz_center = NZ // 2

    metadata = {
        'case': case_key,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'beta': beta,
        'g': g,
        'n_photons': int(repeat_photons * n_repeats),
        'num_repeats': n_repeats,
        'repeat_photons': repeat_photons,
        'repeat_seeds': np.asarray(replicate_seeds, dtype=np.int64),
        'seed': -1 if seed is None else int(seed),
        'sampler': 'rqmc',
        'rqmc_dimensions': RQMC_DIMENSIONS,
        'rqmc_mode': 'launch_only',
        'aggregation': 'mean_of_independent_scrambles',
        'Q_volume': float(Q_volume),
        'total_power': float(total_power),
        'raw_benchmark': True,
        'postprocessed': False,
        'tally_method': 'exact_voxel_traversal',
        'G_center': float(mean_field[ix_center, iy_center, iz_center]),
        'G_max': float(mean_field.max()),
        'G_min': float(mean_field.min()),
        'std_center': float(std_field[ix_center, iy_center, iz_center]),
        'std_mean': float(std_field.mean()),
        'std_max': float(std_field.max()),
    }

    save_aggregate_benchmark(
        case_key,
        mean_field,
        std_field,
        output_dir,
        output_suffix,
        metadata,
    )
    plot_results(mean_field, case_key, output_dir, output_suffix=output_suffix)
    return mean_field, std_field


def plot_results(G_field, case_key, output_dir='MC3D_Results', output_suffix='_HighStats'):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        
        ix_center = NX // 2
        iy_center = NY // 2
        iz_center = NZ // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        im = ax.imshow(G_field[:, :, iz_center].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Case {case_key}: G(x,y,0.5)')
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.imshow(G_field[:, iy_center, :].T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Case {case_key}: G(x,0.5,z)')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 0]
        x_vals = np.linspace(0, 1, NX)
        ax.plot(x_vals, G_field[:, iy_center, iz_center], 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('G(x, 0.5, 0.5)')
        ax.set_title('Centerline Profile')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = (
            f"Case {case_key} Statistics (HIGH STATS)\n"
            f"\n"
            f"G_center:  {G_field[ix_center, iy_center, iz_center]:.4f}\n"
            f"G_max:     {G_field.max():.4f}\n"
            f"G_min:     {G_field.min():.4e}\n"
            f"\n"
            f"Parameters:\n"
            f"κ = {CASE_CONFIGS[case_key]['kappa']}\n"
            f"σs = {CASE_CONFIGS[case_key]['sigma_s']}\n"
            f"g = {CASE_CONFIGS[case_key]['g']}\n"
            f"\n"
            f"High Statistics:\n"
            f"200 Million photons\n"
            f"Raw benchmark field"
        )
        ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'FMC_G_3D_Case{case_key}_FIXED{output_suffix}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FMC 3D RTE Solver (HIGH STATS)')
    parser.add_argument('--case', type=str, choices=['B', 'C', 'all'], default='all')
    parser.add_argument('--output-dir', type=str, default='MC3D_Results')
    parser.add_argument('--n-photons', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output-suffix', type=str, default='_HighStats')
    parser.add_argument('--sampler', type=str, choices=['mc', 'rqmc'], default='mc')
    parser.add_argument('--rqmc-repeats', type=int, default=1)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Forward Monte Carlo 3D RTE Solver - HIGH STATISTICS")
    print("="*70)
    print("\nConfiguration:")
    photon_text = (
        f"{args.n_photons:,} per case (override)"
        if args.n_photons is not None
        else "200,000,000 per case"
    )
    print(f"  Photon count:  {photon_text}")
    print("  Progress:      Every 1% (100 batches)")
    print(f"  Output:        {args.output_suffix} suffix")
    print(f"  Sampler:       {args.sampler.upper()}")
    if args.sampler == 'rqmc':
        print(f"  RQMC repeats:  {args.rqmc_repeats}")
    if args.seed is not None:
        print(f"  RNG seed:      {args.seed}")
    print("="*70 + "\n")
    
    if args.case == 'all':
        cases = ['B', 'C']
    else:
        cases = [args.case]
    
    results = {}
    for case_key in cases:
        case_seed = None if args.seed is None else args.seed + ord(case_key)
        if args.sampler == 'rqmc' and args.rqmc_repeats > 1:
            mean_field, std_field = solve_case_rqmc_replicates(
                case_key,
                output_dir=args.output_dir,
                n_photons_override=args.n_photons,
                seed=case_seed,
                output_suffix=args.output_suffix,
                n_repeats=args.rqmc_repeats,
            )
            results[case_key] = mean_field
        else:
            G_field = solve_case(
                case_key,
                args.output_dir,
                n_photons_override=args.n_photons,
                seed=case_seed,
                output_suffix=args.output_suffix,
                sampler=args.sampler,
            )
            results[case_key] = G_field
            plot_results(G_field, case_key, args.output_dir, output_suffix=args.output_suffix)
        print("\n" + "="*70 + "\n")
    
    print("All cases completed!")
    print(f"Results saved in: {args.output_dir}/")
    
    return results


if __name__ == "__main__":
    main()
