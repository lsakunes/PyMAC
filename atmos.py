import numpy as np
from dataclasses import dataclass

# CONSTANTS
EARTH_WIDTH = 6378137
EARTH_HEIGHT = 6356752

widthOverHeight = EARTH_WIDTH / EARTH_HEIGHT
earthWidthPow2 = EARTH_WIDTH * EARTH_WIDTH

ATMOSPHERE_WIDTH = 6378137 + 11000*1
ATMOSPHERE_HEIGHT = 6356752 + 11000*1

DEG_TO_RAD = 3.14159265358979 / 180.0
RAD_TO_DEG = 180.0 / 3.14159265358979

NUM_SCATTER_POINTS = 3

@dataclass
class RGB:
    r: float
    g: float
    b: float

    def __add__(self, other: "RGB") -> "RGB":
        return RGB(self.r + other.r, self.g + other.g, self.b + other.b)

    def __sub__(self, other: "RGB") -> "RGB":
        return RGB(self.r - other.r, self.g - other.g, self.b - other.b)

    def __neg__(self) -> "RGB":
        return RGB(-self.r, -self.g, -self.b)

    def __mul__(self, other) -> "RGB":
        if isinstance(other, RGB):
            return RGB(self.r * other.r, self.g * other.g, self.b * other.b)
        return RGB(self.r * other, self.g * other, self.b * other)

    __rmul__ = __mul__

    def exp(self) -> "RGB":
        return RGB(np.exp(self.r), np.exp(self.g), np.exp(self.b))

    def dot(self, other: "RGB") -> float:
        return self.r * other.r + self.g * other.g + self.b * other.b

    @staticmethod
    def zeros() -> "RGB":
        return RGB(0.0, 0.0, 0.0)

    @staticmethod
    def ones() -> "RGB":
        return RGB(1.0, 1.0, 1.0)

SIGMA = RGB(0.33, 0.78, 1.89)
S_R = 0.17
SCALE_HEIGHT_RAYLEIGH = 2750

PI = 3.14159265358979323846264338327950288419716939937510582

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
    outscatter: RGB
    inscatter: RGB

LUMINANCE_WEIGHTS = RGB(0.2126, 0.7152, 0.0722)

def luminance(rgb: RGB) -> float:
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


def atmosphere_segment_m(pos_km, dir_unit, sun_unit=None):
    """Returns (start_m, end_m, path_kind) or None. path_kind is 'earth' or 'limb'."""
    del sun_unit  # reserved for future use; kept for MAC adapter signature parity
    ray = _mac_ray(pos_km, dir_unit)

    earth_hit = castRayAgainstOblateSpheroidFull(ray, EARTH_WIDTH, EARTH_HEIGHT)
    atmos_hit = castRayAgainstOblateSpheroidFull(ray, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT)

    if earth_hit.collidesFirst and atmos_hit.collidesFirst:
        return atmos_hit.firstPosition, earth_hit.firstPosition, "earth"

    if atmos_hit.collidesFirst and atmos_hit.collidesSecond and not earth_hit.collidesFirst:
        return atmos_hit.firstPosition, atmos_hit.secondPosition, "limb"

    return None


# Returns atmospheric density at a given altitude
# ASSUMES atmosphere is a fixed distance above the surface around the earth
def atmospheric_density(altitude):
    if altitude < 0.0 or altitude > (ATMOSPHERE_HEIGHT - EARTH_HEIGHT):
        return 0.0

    # densityFalloff = 2.718
    altitude0to1 = altitude / (ATMOSPHERE_WIDTH - EARTH_WIDTH)
    # return 10e9 * 1.227 * exp(-altitude0to1 * densityFalloff) * (1 - altitude0to1)
    return np.exp(-altitude / SCALE_HEIGHT_RAYLEIGH)
    # return exp(-altitude0to1 / 0.25) * (1 - altitude0to1)

# Calculate altitude in kilometers at point
def altitude(point):
    pointDirection = point / np.linalg.norm(point)
    pointOnEarth = castRayAgainstOblateSpheroid(Ray(point, -pointDirection), EARTH_WIDTH, EARTH_HEIGHT).position
    return np.linalg.norm(point - pointOnEarth)

    # float3 normalized = float3(point.x / EARTH_WIDTH, point.y / EARTH_WIDTH, point.z / EARTH_HEIGHT);
    # float scale = 1.0 / length(normalized);
    # float3 surface = point * scale;
    # return length(point - surface);
    # return length(point);

# Calculate optical depth
def opticalDepth(rayOrigin, rayDirection, rayLength):
    densitySamplePoint = rayOrigin
    stepSize = rayLength / float(NUM_SCATTER_POINTS)
    opticalDepth = 0.0

    for i in range(NUM_SCATTER_POINTS):
        densitySamplePoint += rayDirection * stepSize
        opticalDepth += atmospheric_density(altitude(densitySamplePoint)) * stepSize

    return opticalDepth

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
        return scatterResult(outscatter=RGB.ones(), inscatter=RGB.zeros())
    viewDirection = viewVector / viewVectorMagnitude
    stepSize = viewVectorMagnitude / float(NUM_SCATTER_POINTS)

    inScatteredLight = RGB.zeros()
    densityAccum = 0
    phase = phase_rayleigh(viewDirection, -sunDirection)
    sigma_r = S_R * SIGMA
    output = np.array([0.0, 0.0, 0.0])

    # iterating through sample points from start to end
    for i in range(NUM_SCATTER_POINTS):
        t = float(i) * stepSize
        # slang: (t == 0 ? epsilon : (t == viewVectorMagnitude ? t - epsilon : t))
        t_sample = epsilon if t == 0.0 else (t - epsilon if t == viewVectorMagnitude else t)
        point = startPoint + t_sample * viewDirection

        # Sun Length calculation
        sunRay = Ray(point, -sunDirection)
        intersectionPoint = castRayAgainstOblateSpheroidFull(sunRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT).secondPosition
        sunRayLength = np.linalg.norm(intersectionPoint - point)

        if intersectionPoint[0] == 9.0: output += np.array([1, 0, 0])
        output += np.array([0, intersectionPoint[1], intersectionPoint[2]]) * 0.0000001 / NUM_SCATTER_POINTS
        
        #In and out scattering
        localDensity = atmospheric_density(altitude(point))
        densityAccum += localDensity * stepSize

        if not castRayAgainstOblateSpheroidFull(sunRay, EARTH_WIDTH, EARTH_HEIGHT).collidesFirst:
            sunRayOpticalDepth = opticalDepth(point, -sunDirection, sunRayLength)
            viewRayOpticalDepth = opticalDepth(point, viewDirection, t)

            transmittance = (-(sunRayOpticalDepth + viewRayOpticalDepth) * sigma_r).exp()
            scatteredSunIntoViewRay = RGB.ones() * localDensity * transmittance * phase * sigma_r
            inScatteredLight += scatteredSunIntoViewRay * stepSize

    # float3 outScatterFactor = exp(-densityAccum * 0.001 * sigma_r); // sigma_r *
    outScatterFactor = (-densityAccum * 0.001 * sigma_r).exp()
    return scatterResult(outscatter=outScatterFactor, inscatter=inScatteredLight)
    # return scatterResult(outScatterFactor, inScatteredLight);
    # return rayBrightness + inScatteredLight; #densityAccum * 0.0001;
    # return densityAccum * 0.0001;
    # return outScatterFactor;
    # return rayBrightness + inScatteredLight;
    # * (1 - rayBrightness); 
    #densityAccum * (1-rayBrightness); 
    #startKm * 0.0001;
    # return rayBrightness * outScatterFactor + inScatteredLight;
