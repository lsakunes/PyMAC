module atmos;
import "../src/renderer/vulkan/constants.slang";
import "../src/renderer/vulkan/intersection.slang";

// CONSTANTS
static const int NUM_SCATTER_POINTS = 20;

static const float3 SIGMA = float3(0.33, 0.78, 1.89);
static const float S_R = 0.17;

static const float SCALE_HEIGHT_RAYLEIGH = 2750;



// Returns atmospheric density at a given altitude
// ASSUMES atmosphere is a fixed distance above the surface around the earth
float atmospheric_density(float altitude) {
    if (altitude < 0.0 || altitude > (ATMOSPHERE_HEIGHT - EARTH_HEIGHT)) {
        return 0.0;
    }

    // float densityFalloff = 2.718;
    float altitude0to1 = altitude / (ATMOSPHERE_WIDTH - EARTH_WIDTH);
    // return 10e9 * 1.227 * exp(-altitude0to1 * densityFalloff) * (1 - altitude0to1);
    return exp(-altitude / SCALE_HEIGHT_RAYLEIGH);
    // return exp(-altitude0to1 / 0.25) * (1 - altitude0to1);
}

// Calculate altitude in kilometers at point
public float altitude(float3 point) {
    float3 pointDirection = normalize(point);
    float3 pointOnEarth = castRayAgainstOblateSpheroid(Ray(point, -pointDirection), EARTH_WIDTH, EARTH_HEIGHT).position;
    return length(point - pointOnEarth);

    // float3 normalized = float3(point.x / EARTH_WIDTH, point.y / EARTH_WIDTH, point.z / EARTH_HEIGHT);
    // float scale = 1.0 / length(normalized);
    // float3 surface = point * scale;
    // return length(point - surface);
    // return length(point);
}

// Calculate optical depth
float opticalDepth(float3 rayOrigin, float3 rayDirection, float rayLength) {
    float3 densitySamplePoint = rayOrigin;
    float stepSize = rayLength / float(NUM_SCATTER_POINTS);
    float opticalDepth = 0.0;

    for (int i = 0; i < NUM_SCATTER_POINTS; i++) {
        densitySamplePoint += rayDirection * stepSize;
        opticalDepth += atmospheric_density(altitude(densitySamplePoint)) * stepSize;
    }

    return opticalDepth;
}

// Describes what fraction of light scatters toward the viewer based on the angle
// between the sun and view direction.
public float phase_rayleigh(float3 view_dir, float3 rayDir) {
    float cos_theta = dot(rayDir, view_dir);
    return (3.0 / (16.0 * 3.14159265358979)) * (1.0 + cos_theta * cos_theta);
}

// returns the final RGB brightness at startPoint after the incoming ray is dimmed by out-scattering
// and sunlight scattered into the ray is added (in-scattering)
// inputs should be in meters
// ray brightness is the brightness of the ray at the start point
// this function returns the brightness of the ray at the end point after going through the atmopshere
public scatterResult calculateAtmosphericDimming(float3 startPoint, float3 endPoint, float3 sunDirection) {
    float epsilon = 10;
    float3 viewVector = endPoint - startPoint;
    float viewVectorMagnitude = length(viewVector);
    float3 viewDirection = viewVector / viewVectorMagnitude;
    float stepSize = viewVectorMagnitude / float(NUM_SCATTER_POINTS);

    float3 inScatteredLight = float3(0.0);
    float densityAccum = 0;
    float phase = phase_rayleigh(viewDirection, -sunDirection);
    float3 sigma_r = S_R * SIGMA;
    float3 output = float3(0);
    // iterating through sample points from start to end
    for (int i = 0; i < NUM_SCATTER_POINTS; i++) {
        float t = float(i) * stepSize;
        float3 point = startPoint + (t == 0 ? epsilon : (t == viewVectorMagnitude ? t - epsilon : t)) * viewDirection; 
        // Sun Length calculation
        Ray sunRay = { point, -sunDirection };
        float3 intersectionPoint = castRayAgainstOblateSpheroidFull(sunRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT).secondPosition;
        float sunRayLength = length(intersectionPoint - point);
        if (intersectionPoint.x == 9.0) output += float3(1, 0, 0);
        output += float3(0, intersectionPoint.y, intersectionPoint.z) * 0.0000001 / NUM_SCATTER_POINTS;
        // In and out scattering
        float localDensity = atmospheric_density(altitude(point));
        densityAccum += localDensity * stepSize;

        if (!castRayAgainstOblateSpheroidFull(sunRay, EARTH_WIDTH, EARTH_HEIGHT).collidesFirst) {
            float sunRayOpticalDepth = opticalDepth(point, -sunDirection, sunRayLength);
            float viewRayOpticalDepth = opticalDepth(point, viewDirection, t);

            float3 transmittance = exp(-(sunRayOpticalDepth + viewRayOpticalDepth) * sigma_r);
            float3 scatteredSunIntoViewRay = float3(1.0) * localDensity * transmittance * phase * sigma_r;
            inScatteredLight += scatteredSunIntoViewRay * stepSize;
        }
    }

    return scatterResult(1, 1);

    // float3 outScatterFactor = exp(-densityAccum * 0.001 * sigma_r); // sigma_r *
    // return scatterResult(outScatterFactor, inScatteredLight);
    // return rayBrightness + inScatteredLight; //densityAccum * 0.0001;
    // return densityAccum * 0.0001;
    // return outScatterFactor;
    // return rayBrightness + inScatteredLight;
    // * (1 - rayBrightness); 
    //densityAccum * (1-rayBrightness); 
    //startKm * 0.0001;
    // return rayBrightness * outScatterFactor + inScatteredLight;
}