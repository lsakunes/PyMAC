"""Taichi port of atmos.py's per-ray math, for GPU-parallel rendering.

Mirrors the functions in atmos.py one-for-one so the two can be diffed against
each other for correctness (see GPU_ACCELERATION.md). Dataclasses become plain
tuple returns and NumPy arrays become ti.math.vec3, since Taichi kernels/funcs
can't allocate Python objects. Everything here runs in ti.f32 (GPU-native),
while atmos.py stays in float64 on the CPU -- expect small numeric drift
between the two when validating.
"""

import numpy as np
import taichi as ti
from pathlib import Path

vec3 = ti.math.vec3

# CONSTANTS -- kept as plain Python floats; Taichi captures these as
# compile-time constants when a ti.func/ti.kernel referencing them is compiled.
EARTH_WIDTH = 6378137.0
EARTH_HEIGHT = 6356752.0

ATMOSPHERE_WIDTH = 6378137.0 + 11000.0
ATMOSPHERE_HEIGHT = 6356752.0 + 11000.0
ATMOSPHERE_THICKNESS_M = ATMOSPHERE_HEIGHT - EARTH_HEIGHT

NUM_SCATTER_POINTS = 20

SIGMA = vec3(0.33, 0.78, 1.89)
S_R = 0.17

LUMINANCE_WEIGHTS = vec3(0.2126, 0.7152, 0.0722)

_SCATTER_PROFILE_PATH = Path(__file__).resolve().parent / "scatter_profile.npz"

# Populated by load_scatter_profile(); a 1D field of RGB scattering
# coefficients sampled at a fixed altitude step, uploaded once to the GPU.
_MAX_ALT_SAMPLES = 256
beta_field = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_ALT_SAMPLES)
alt_step = ti.field(ti.f32, shape=())
alt_n = ti.field(ti.i32, shape=())


def load_scatter_profile(path: Path = _SCATTER_PROFILE_PATH) -> None:
    """Load scatter_profile.npz (built by MAC.build_scatter_profile) onto the GPU.

    Assumes alt_km is evenly spaced (true for np.arange-built profiles), so
    scattering_coefficient_rgb() can index directly instead of a binary search.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Call build_scatter_profile() before rendering."
        )
    data = np.load(path)
    alt_km = np.asarray(data["alt_km"], dtype=np.float64)
    beta_rgb = np.asarray(data["beta_rgb"], dtype=np.float64)
    n = len(alt_km)
    if n > _MAX_ALT_SAMPLES:
        raise ValueError(
            f"scatter_profile.npz has {n} altitude samples; "
            f"_MAX_ALT_SAMPLES ({_MAX_ALT_SAMPLES}) must be raised to match."
        )
    padded = np.zeros((_MAX_ALT_SAMPLES, 3), dtype=np.float32)
    padded[:n] = beta_rgb
    beta_field.from_numpy(padded)
    alt_step[None] = float(alt_km[1] - alt_km[0])
    alt_n[None] = n


@ti.func
def scattering_coefficient_rgb(altitude_m: ti.f32) -> vec3:
    """RGB volume scattering coefficient at altitude (meters); GPU-side lerp
    into the table loaded by load_scatter_profile()."""
    result = vec3(0.0, 0.0, 0.0)
    if 0.0 <= altitude_m <= ATMOSPHERE_THICKNESS_M:
        alt_km = altitude_m / 1000.0
        idx_f = alt_km / alt_step[None]
        n = alt_n[None]
        i0 = ti.max(0, ti.min(ti.cast(ti.floor(idx_f), ti.i32), n - 1))
        i1 = ti.max(0, ti.min(i0 + 1, n - 1))
        frac = ti.max(0.0, ti.min(1.0, idx_f - ti.cast(i0, ti.f32)))
        result = beta_field[i0] * (1.0 - frac) + beta_field[i1] * frac
    return result


@ti.func
def _convert_ray_to_sphere_space(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    w_over_h = width / height
    sphere_origin = vec3(origin.x, origin.y, origin.z * w_over_h)
    sphere_dir = vec3(direction.x, direction.y, direction.z * w_over_h)
    return sphere_origin, sphere_dir.normalized()


@ti.func
def _convert_pos_to_ellipsoid_space(pos: vec3, width: ti.f32, height: ti.f32) -> vec3:
    return vec3(pos.x, pos.y, pos.z / (width / height))


@ti.func
def cast_ray_against_oblate_spheroid_full(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    """Both intersections of `origin + t*direction` with the oblate spheroid.

    Returns (first_pos, collides_first, second_pos, collides_second) -- the
    tuple equivalent of atmos.py's ComplexRaycastResult.
    """
    sphere_origin, sphere_dir = _convert_ray_to_sphere_space(origin, direction, width, height)

    a = sphere_dir.dot(sphere_dir)
    b = 2.0 * sphere_origin.dot(sphere_dir)
    c = sphere_origin.dot(sphere_origin) - width * width
    determinant = (-4.0 * c * a) + (b * b)

    first_pos = vec3(9.0, 9.0, 9.0)
    collides_first = False
    second_pos = vec3(0.0, 0.0, 0.0)
    collides_second = False

    if determinant >= 0.0:
        sqrt_det = ti.sqrt(determinant)
        two_a = 2.0 * a
        small_t = (-b - sqrt_det) / two_a
        large_t = (-b + sqrt_det) / two_a

        first_pos = vec3(0.0, 0.0, 0.0)
        second_pos = vec3(0.0, 0.0, 0.0)

        if small_t >= 0.0:
            hit = sphere_origin + sphere_dir * small_t
            first_pos = _convert_pos_to_ellipsoid_space(hit, width, height)
            collides_first = True
        if large_t >= 0.0:
            hit = sphere_origin + sphere_dir * large_t
            second_pos = _convert_pos_to_ellipsoid_space(hit, width, height)
            collides_second = True

    return first_pos, collides_first, second_pos, collides_second


@ti.func
def _mac_ray(pos_km: vec3, dir_unit: vec3):
    """MAC traces pos - t*dir; oblate cast uses origin + t*direction."""
    return pos_km * 1000.0, -dir_unit


@ti.func
def earth_hit_km(pos_km: vec3, dir_unit: vec3):
    ray_origin, ray_dir = _mac_ray(pos_km, dir_unit)
    hit_pos, collides, _, _ = cast_ray_against_oblate_spheroid_full(
        ray_origin, ray_dir, EARTH_WIDTH, EARTH_HEIGHT
    )
    return hit_pos / 1000.0, collides


@ti.func
def atmosphere_segment_m(pos_km: vec3, dir_unit: vec3):
    ray_origin, ray_dir = _mac_ray(pos_km, dir_unit)
    earth_hit, _, _, _ = cast_ray_against_oblate_spheroid_full(ray_origin, ray_dir, EARTH_WIDTH, EARTH_HEIGHT)
    atmos_hit, _, _, _ = cast_ray_against_oblate_spheroid_full(ray_origin, ray_dir, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT)
    return atmos_hit, earth_hit


@ti.func
def altitude(point: vec3) -> ti.f32:
    point_dir = point.normalized()
    surface, _, _, _ = cast_ray_against_oblate_spheroid_full(point, -point_dir, EARTH_WIDTH, EARTH_HEIGHT)
    return (point - surface).norm()


@ti.func
def optical_depth(ray_origin: vec3, ray_direction: vec3, ray_length: ti.f32) -> vec3:
    sample_point = ray_origin
    step_m = ray_length / NUM_SCATTER_POINTS
    step_km = step_m / 1000.0
    total = vec3(0.0, 0.0, 0.0)
    for _ in range(NUM_SCATTER_POINTS):
        sample_point = sample_point + ray_direction * step_m
        total += scattering_coefficient_rgb(altitude(sample_point)) * step_km
    return total


@ti.func
def phase_rayleigh(view_dir: vec3, ray_dir: vec3) -> ti.f32:
    cos_theta = ray_dir.dot(view_dir)
    return (3.0 / (16.0 * np.pi)) * (1.0 + cos_theta * cos_theta)


@ti.func
def calculate_atmospheric_dimming(start_point: vec3, end_point: vec3, sun_direction: vec3):
    """Returns (outscatter, inscatter) -- tuple equivalent of scatterResult."""
    epsilon = 10.0
    view_vector = end_point - start_point
    view_mag = view_vector.norm()

    outscatter = vec3(1.0, 1.0, 1.0)
    inscatter = vec3(0.0, 0.0, 0.0)

    if view_mag > 0.0:
        view_dir = view_vector / view_mag
        step_m = view_mag / NUM_SCATTER_POINTS
        step_km = step_m / 1000.0
        tau_view_accum = vec3(0.0, 0.0, 0.0)
        phase = phase_rayleigh(view_dir, -sun_direction)
        in_scattered = vec3(0.0, 0.0, 0.0)
        sun_ray_dir = -sun_direction

        for i in range(NUM_SCATTER_POINTS):
            t = ti.cast(i, ti.f32) * step_m
            t_sample = t
            if t == 0.0:
                t_sample = epsilon
            elif t == view_mag:
                t_sample = t - epsilon
            point = start_point + t_sample * view_dir

            _, _, atmos_second, _ = cast_ray_against_oblate_spheroid_full(
                point, sun_ray_dir, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT
            )
            sun_ray_length = (atmos_second - point).norm()

            _, earth_collides_first, _, _ = cast_ray_against_oblate_spheroid_full(
                point, sun_ray_dir, EARTH_WIDTH, EARTH_HEIGHT
            )

            beta_rgb = scattering_coefficient_rgb(altitude(point)) * S_R
            tau_view_accum += beta_rgb * step_km

            if not earth_collides_first:
                sun_optical_depth = optical_depth(point, sun_ray_dir, sun_ray_length)
                view_optical_depth = optical_depth(point, view_dir, t)
                transmittance = ti.exp(-(sun_optical_depth + view_optical_depth))
                in_scattered += transmittance * phase * beta_rgb * step_km

        outscatter = ti.exp(-tau_view_accum)
        inscatter = in_scattered

    return outscatter, inscatter


@ti.func
def luminance(rgb: vec3) -> ti.f32:
    return rgb.dot(LUMINANCE_WEIGHTS)
