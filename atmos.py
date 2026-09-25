import numpy as np
from dataclasses import dataclass
from pathlib import Path

# CONSTANTS
EARTH_WIDTH = 6378137
EARTH_HEIGHT = 6356752

widthOverHeight = EARTH_WIDTH / EARTH_HEIGHT
earthWidthPow2 = EARTH_WIDTH * EARTH_WIDTH

# +11 km shell (matches MAC atmosphere=11); MSIS profile is 0–11 km
ATMOSPHERE_WIDTH = 6378137 + 11000*1
ATMOSPHERE_HEIGHT = 6356752 + 11000*1
ATMOSPHERE_THICKNESS_M = ATMOSPHERE_HEIGHT - EARTH_HEIGHT

DEG_TO_RAD = 3.14159265358979 / 180.0
RAD_TO_DEG = 180.0 / 3.14159265358979

NUM_SCATTER_POINTS = 3


SIGMA = np.array([0.33, 0.78, 1.89])
S_R = 0.17
SCALE_HEIGHT_RAYLEIGH = 2750

PI = 3.14159265358979323846264338327950288419716939937510582

_SCATTER_PROFILE_PATH = Path(__file__).resolve().parent / "scatter_profile.npz"
_alt_km = None
_beta_rgb = None
_beta_sea_level = None


def reload_scatter_profile():
    """Clear cached table so the next lookup reloads scatter_profile.npz from disk."""
    global _alt_km, _beta_rgb, _beta_sea_level
    _alt_km = None
    _beta_rgb = None
    _beta_sea_level = None


def _load_scatter_profile():
    """Load precomputed alt_km -> beta_rgb. No nrlmsise00 at render time."""
    global _alt_km, _beta_rgb, _beta_sea_level
    if _alt_km is not None:
        return
    if not _SCATTER_PROFILE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {_SCATTER_PROFILE_PATH.name}. "
            "Call build_scatter_profile() from MAC before rendering."
        )
    data = np.load(_SCATTER_PROFILE_PATH)
    _alt_km = np.asarray(data["alt_km"], dtype=np.float64)
    _beta_rgb = np.asarray(data["beta_rgb"], dtype=np.float64)
    _beta_sea_level = _beta_rgb[0].copy()


def scattering_coefficient_rgb(altitude_m):
    """RGB volume scattering coefficient at altitude (meters) from MSIS table."""
    _load_scatter_profile()
    if altitude_m < 0.0 or altitude_m > ATMOSPHERE_THICKNESS_M:
        return np.zeros(3, dtype=np.float64)
    alt_km = altitude_m / 1000.0
    return np.array(
        [
            np.interp(alt_km, _alt_km, _beta_rgb[:, 0]),
            np.interp(alt_km, _alt_km, _beta_rgb[:, 1]),
            np.interp(alt_km, _alt_km, _beta_rgb[:, 2]),
        ],
        dtype=np.float64,
    )

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

@dataclass
class SimpleRaycastResult:
    collides: bool
    position: np.ndarray

@dataclass
class ComplexRaycastResult:
    firstPosition: np.ndarray
    collidesFirst: bool

    secondPosition: np.ndarray
    collidesSecond: bool

@dataclass
class scatterResult:
    outscatter: np.array
    inscatter: np.array

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

def luminance(rgb: np.array) -> float:
    return rgb.dot(LUMINANCE_WEIGHTS)

# Calculates both collision positions for the raycast
def castRayAgainstOblateSpheroidFull(ray, width, height):
    sphereSpaceRay = convertRayToSphereSpace(ray, width, height)

    # The pieces of the quadratic formula
    a = np.dot(sphereSpaceRay.direction, sphereSpaceRay.direction)
    b = 2.0 * np.dot(sphereSpaceRay.origin, sphereSpaceRay.direction)
    c = np.dot(sphereSpaceRay.origin, sphereSpaceRay.origin) - (width * width)

    determinant = ((-4.0 * c) * a) + (b * b)

    # Check if the ray misses the ellipsoid
    if determinant < 0.0:
        return ComplexRaycastResult(firstPosition=np.array([9.0, 9.0, 9.0]), collidesFirst=False, secondPosition=np.array([0.0, 0.0, 0.0]), collidesSecond=False)


    # This will give us the smallest value of t, and this is almost always what we want,
    # but we need to throw this out in the case where t is negative. This would be the case
    # where the earth is behind the camera, and we don't want earth to be visible in that case.
    # We don't know if a is positive or negative, so we need to evaluate the whole thing to determine t's sign.
    sqrtDeterminant = np.sqrt(determinant)
    twoA = 2.0 * a

    smallT = (-b - sqrtDeterminant) / twoA
    largeT = (-b + sqrtDeterminant) / twoA

    smallTPosition = np.array([0.0, 0.0, 0.0])
    largeTPosition = np.array([0.0, 0.0, 0.0])

    if smallT >= 0.0:
        # This is in sphere space still so we need to convert it back before returning it
        sphereSpaceHitLocation = (sphereSpaceRay.direction * smallT) + sphereSpaceRay.origin

        smallTPosition = convertPosToEllipsoidSpace(sphereSpaceHitLocation, width, height)

    if largeT >= 0.0:
        # This is in sphere space still so we need to convert it back before returning it
        sphereSpaceHitLocation = (sphereSpaceRay.direction * largeT) + sphereSpaceRay.origin

        largeTPosition = convertPosToEllipsoidSpace(sphereSpaceHitLocation, width, height)

    return ComplexRaycastResult(
        firstPosition=smallTPosition,
        collidesFirst=smallT >= 0.0,
        secondPosition=largeTPosition,
        collidesSecond=largeT >= 0.0)

def castRayAgainstOblateSpheroid(ray, width, height):
    raycastResult = castRayAgainstOblateSpheroidFull(ray, width, height)
    return SimpleRaycastResult(collides=raycastResult.collidesFirst, position=raycastResult.firstPosition)

# Coordinate Conversions
def convertPosToSphereSpace(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    return np.array([pos[0], pos[1], pos[2] * (width / height)])

def convertPosToEllipsoidSpace(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    return np.array([pos[0], pos[1], pos[2] / (width / height)])

def convertRayToSphereSpace(ray: Ray, width: float, height: float) -> Ray:
    widthDivHeight = width / height
    vector = np.array([ray.direction[0], ray.direction[1], ray.direction[2] * widthDivHeight])
    return Ray(
        np.array([ray.origin[0], ray.origin[1], ray.origin[2] * widthDivHeight]),
        vector / np.linalg.norm(vector)
    )

def convertRayToEllipsoidSpace(ray: Ray, width: float, height: float) -> Ray:
    widthDivHeight = width / height
    vector = np.array([ray.direction[0], ray.direction[1], ray.direction[2] / widthDivHeight])
    return Ray(
        np.array([ray.origin[0], ray.origin[1], ray.origin[2] / widthDivHeight]),
        vector / np.linalg.norm(vector)
    )

def _mac_ray(pos_km, dir_unit):
    """MAC traces pos - t*dir; oblate cast uses origin + t*direction."""
    return Ray(pos_km * 1000.0, -dir_unit)


def earth_hit_km(pos_km, dir_unit):
    """Returns (surface_point_km, distance_km) or None."""
    ray = _mac_ray(pos_km, dir_unit)
    hit = castRayAgainstOblateSpheroidFull(ray, EARTH_WIDTH, EARTH_HEIGHT)
    if not hit.collidesFirst:
        return None
    surface_m = hit.firstPosition
    return surface_m / 1000.0


def atmosphere_segment_m(pos_km, dir_unit):
    ray = _mac_ray(pos_km, dir_unit)

    earth_hit = castRayAgainstOblateSpheroidFull(ray, EARTH_WIDTH, EARTH_HEIGHT)
    atmos_hit = castRayAgainstOblateSpheroidFull(ray, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT)

    return atmos_hit.firstPosition, earth_hit.firstPosition


# Scalar relative scattering strength (0–1 vs sea level) from MSIS table.
# ASSUMES atmosphere is a fixed distance above the surface around the earth
def atmospheric_density(altitude_m):
    beta = scattering_coefficient_rgb(altitude_m)
    _load_scatter_profile()
    denom = float(np.mean(_beta_sea_level))
    if denom <= 0.0:
        return 0.0
    return float(np.mean(beta) / denom)

# Calculate altitude in meters at point
def altitude(point):
    pointDirection = point / np.linalg.norm(point)
    pointOnEarth = castRayAgainstOblateSpheroid(Ray(point, -pointDirection), EARTH_WIDTH, EARTH_HEIGHT).position
    return np.linalg.norm(point - pointOnEarth)

    # float3 normalized = float3(point.x / EARTH_WIDTH, point.y / EARTH_WIDTH, point.z / EARTH_HEIGHT);
    # float scale = 1.0 / length(normalized);
    # float3 surface = point * scale;
    # return length(point - surface);
    # return length(point);

# RGB optical depth along a ray segment (beta = S_R*SIGMA*(n_w/N_w0) per km;
# path lengths are in meters, so convert step to km — matches legacy *0.001).
def opticalDepth(rayOrigin, rayDirection, rayLength):
    densitySamplePoint = rayOrigin
    stepSize_m = rayLength / float(NUM_SCATTER_POINTS)
    stepSize_km = stepSize_m / 1000.0
    opticalDepthRgb = np.zeros(3, dtype=np.float64)

    for i in range(NUM_SCATTER_POINTS):
        densitySamplePoint += rayDirection * stepSize_m
        opticalDepthRgb += scattering_coefficient_rgb(altitude(densitySamplePoint)) * stepSize_km

    return opticalDepthRgb

# Describes what fraction of light scatters toward the viewer based on the angle
# between the sun and view direction.
def phase_rayleigh(view_dir, rayDir):
    cos_theta = np.dot(rayDir, view_dir)
    return (3.0 / (16.0 * 3.14159265358979)) * (1.0 + cos_theta * cos_theta)

# returns the final RGB brightness at startPoint after the incoming ray is dimmed by out-scattering
# and sunlight scattered into the ray is added (in-scattering)
# inputs should be in meters
# ray brightness is the brightness of the ray at the start point
# this function returns the brightness of the ray at the end point after going through the atmopshere
def calculateAtmosphericDimming(startPoint, endPoint, sunDirection):
    epsilon = 10
    viewVector = endPoint - startPoint
    viewVectorMagnitude = np.linalg.norm(viewVector)
    if viewVectorMagnitude <= 0.0:
        return scatterResult(outscatter=np.array([1,1,1]), inscatter=np.array([0,0,0]))
    viewDirection = viewVector / viewVectorMagnitude
    stepSize_m = viewVectorMagnitude / float(NUM_SCATTER_POINTS)
    stepSize_km = stepSize_m / 1000.0

    inScatteredLight = np.array([0.0, 0.0, 0.0])
    tau_view_accum_rgb = np.zeros(3, dtype=np.float64)
    phase = phase_rayleigh(viewDirection, -sunDirection)
    output = np.array([0.0, 0.0, 0.0])

    # iterating through sample points from start to end
    for i in range(NUM_SCATTER_POINTS):
        t = float(i) * stepSize_m
        # slang: (t == 0 ? epsilon : (t == viewVectorMagnitude ? t - epsilon : t))
        t_sample = epsilon if t == 0.0 else (t - epsilon if t == viewVectorMagnitude else t)
        point = startPoint + t_sample * viewDirection

        # Sun Length calculation
        sunRay = Ray(point, -sunDirection)
        intersectionPoint = castRayAgainstOblateSpheroidFull(sunRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT).secondPosition
        sunRayLength = np.linalg.norm(intersectionPoint - point)

        if intersectionPoint[0] == 9.0: output += np.array([1, 0, 0])
        output += np.array([0, intersectionPoint[1], intersectionPoint[2]]) * 0.0000001 / NUM_SCATTER_POINTS

        # In and out scattering — beta_rgb from MSIS N2/O2/Ar table (alt -> strength)
        beta_rgb = scattering_coefficient_rgb(altitude(point)) * S_R
        tau_view_accum_rgb += beta_rgb * stepSize_km

        if not castRayAgainstOblateSpheroidFull(sunRay, EARTH_WIDTH, EARTH_HEIGHT).collidesFirst:
            sunRayOpticalDepth = opticalDepth(point, -sunDirection, sunRayLength)
            viewRayOpticalDepth = opticalDepth(point, viewDirection, t)

            transmittance = np.exp(-(sunRayOpticalDepth + viewRayOpticalDepth))
            scatteredSunIntoViewRay = transmittance * phase * beta_rgb
            inScatteredLight += scatteredSunIntoViewRay * stepSize_km

    # beta is per km (S_R*SIGMA at sea level); no legacy *0.001 density fudge.
    # Retune S_R (+ re-run MAC so build_scatter_profile refreshes the table) if limb/disk is too bright/dim.
    outScatterFactor = np.exp(-tau_view_accum_rgb)
    return scatterResult(outscatter=outScatterFactor, inscatter=inScatteredLight)
