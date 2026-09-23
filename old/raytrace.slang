import atmos;
import "../src/renderer/vulkan/constants.slang";
import "../src/renderer/vulkan/intersection.slang";

//////////
// Types
//////////
struct CameraData {
    float3 position;
    float3 forward;
    float3 up;
    float3 right;

    float focalLength;
    float pixelSize;

    int2 resolution;
}

struct RayBounceResult {
    float3 earthColor;
    float4 earthMaterial;
    float3 earthNormal;
    Ray bounceRay[rayBounceCount];
    float3 bounceRayColor[rayBounceCount];
    float bounceCosine[rayBounceCount];
    float3 specularReflectionVector[rayBounceCount];
}


/////////////
// Constants
/////////////



// Colors
static const float3 spaceColor = float3(0.0, 0.0, 0.0);
static const float3 starColor = float3(0, 0, 0);

// Scene
static const float3 sunVector = float3(0.0, 1.0, 0);

static const float earthColorLightMultiplier = 1.5;
static const float earthSpecularCoefficient = 0.5;
static const float earthSpecularExponent = 10;

static const float shadowAcneCompensationValue = 10;
static const int rayBounceCount = 10;

// Pixel Offset
static const float2 pixelOffsets[] = {
    float2(-0.5, -0.5), float2(0.0, -0.5), float2(0.5, -0.5),
    float2(-0.5, 0.0), float2(0.0, 0.0), float2(0.5, 0.0),
    float2(-0.5, 0.5), float2(0.0, 0.5), float2(0.5, 0.5)
};

///////////
// Inputs
///////////
[[vk_push_constant]]
CameraData camera;

[vk::binding(0, 0)]
Texture2D<float4> earthProjection;
[vk::binding(1, 0)]
SamplerState earthProjectionSampler;

[vk::binding(2, 0)]
Texture2D<float4> earthMaterial;
[vk::binding(3, 0)]
SamplerState earthMaterialSampler;

[vk::binding(4, 0)]
[vk::image_format("rgba8")]
RWTexture2D<float4> outputImage;

float pcgHashRandom(const uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint big_number = (word >> 22u) ^ word;

    return (float)big_number / 4294967296.0;
}

uint generatePcgHashSeed(const uint2 threadId, const uint funNumber) {
    uint seed = (threadId.x + 3319) * (threadId.y + 103) * funNumber;
    return (seed ^ 61) ^ (seed >> 16);
}

float3 getRandomUnitVector(const uint2 threadId, const uint seed) {
    let theta = (2.0 * PI) * pcgHashRandom(generatePcgHashSeed(threadId, seed * 47));
    let z = (2.0 * pcgHashRandom(generatePcgHashSeed(threadId, seed * 53))) - 1.0;

    let horizontalFactor = sqrt(1.0 - (z * z));

    return normalize(float3(
        horizontalFactor * cos(theta),
        horizontalFactor * sin(theta),
        z
    ));
}

[shader("compute")]
[numthreads(1,1,1)]
void raytracerMain(const uint3 threadId: SV_DispatchThreadID) {
    float3 colorAverage = float3(0.0);

    for(int i = 0; i < pixelOffsets.getCount(); ++i) {
        let pixelLightRayDirection = getCameraPixelLightRayDirection(threadId.xy, pixelOffsets[i]);

        let cameraRay = Ray(camera.position, pixelLightRayDirection);
        let raycastResult = castRayAgainstOblateSpheroid(cameraRay, EARTH_WIDTH, EARTH_HEIGHT);

        if (!raycastResult.collides) {
            colorAverage += getSpaceColor(threadId.xy, cameraRay);
        }
        else {
            let cameraRayAtmosphereRaycastResult = castRayAgainstOblateSpheroid(cameraRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT);

            let earthBounceResult = bounceRayOnEarth(Ray(raycastResult.position, -pixelLightRayDirection), threadId.xy);

            scatterResult atmosphereCalcs =
                                           !cameraRayAtmosphereRaycastResult.collides ? scatterResult(0,0) :
                calculateAtmosphericDimming(
                    cameraRayAtmosphereRaycastResult.position,
                    raycastResult.position,
                    sunVector); //earthBounceResult.earthColor.xyz);

            var earthColorAverage = float3(0.0);
            for (int i = 0; i < rayBounceCount; ++i) {
                // earthColorAverage += atmosphereCalcs.outscatter * getReflectedRayColor(
                //                                                       earthBounceResult.earthMaterial,
                //                                                       earthBounceResult.earthNormal,
                //                                                       earthBounceResult.bounceRayColor[i],
                //                                                       earthBounceResult.bounceCosine[i],
                //                                                       cameraRay.direction,
                //                                                       earthBounceResult.specularReflectionVector[i]);
                earthColorAverage += atmosphereCalcs.inscatter + atmosphereCalcs.outscatter * earthBounceResult.earthColor.xyz * dot(normalize(raycastResult.position), normalize(-sunVector));
            }
            earthColorAverage /= rayBounceCount;
            colorAverage += earthColorAverage;
        }
    }

    outputImage[threadId.xy] = float4((colorAverage / pixelOffsets.getCount()), 1.0);
}

// Calculates the direction of the vector coming from the camera going through the pixel
// in the image plane if the image plane was in front of the camera instead of behind it.
// The origin of the ray is the camera position.
float3 getCameraPixelLightRayDirection(const int2 pixelCoords, const float2 pixelOffset) {
    // Pixel coordinates with (0, 0) being in the center and negative in the top left
    let normalizedPixelCoordinate = float2(
        float(pixelCoords.x) - (float(camera.resolution.x) / 2.0) + pixelOffset.x,
        (float(camera.resolution.y) - float(pixelCoords.y)) - (float(camera.resolution.y) / 2.0) + pixelOffset.y
    );

    // Offset on the image plane using world units
    let pixelImgPlaneCenterOffset = (normalizedPixelCoordinate + 0.5) * camera.pixelSize;

    return
        (camera.forward * camera.focalLength) +
        (camera.right * pixelImgPlaneCenterOffset.x) +
        (camera.up * pixelImgPlaneCenterOffset.y);
}

// Checks if the ray hits the earth, and if so, calculates the closest position at which they collide
float findSphericalLatitudeOfOblateSpheroidPoint(float3 point, float width, float height) {
    let pointSphereSpace = convertPosToSphereSpace(point, width, height);

    return asin(-pointSphereSpace.z / width);
}

float findSphericalLongitudeOfOblateSpheroidPoint(float3 point) {
    return atan2(point.x, -point.y);
}

float2 findTextureCoordsFromSphericalAngles(float latitude, float longitude) {
    return float2(
        (longitude + PI) / (2.0 * PI),
        (latitude + (PI / 2)) / PI
    );
}

RayBounceResult bounceRayOnEarth(const Ray ray, const int2 threadId) {
    let earthNormal = getOblateSpheroidNormalVector(ray.origin, EARTH_WIDTH, EARTH_HEIGHT);

    let textureCoords = findTextureCoordsFromSphericalAngles(
               findSphericalLatitudeOfOblateSpheroidPoint(ray.origin, EARTH_WIDTH, EARTH_HEIGHT),
               findSphericalLongitudeOfOblateSpheroidPoint(ray.origin));

    let textureColor = earthProjection.SampleLevel(
        earthProjectionSampler,
        textureCoords,
        0
    ).xyz;

    let materialSample = earthMaterial.SampleLevel(
        earthMaterialSampler,
        textureCoords,
        0
    );

    Ray reflectedRays[rayBounceCount];
    float3 reflectedRayColor[rayBounceCount];
    float reflectedRayCosines[rayBounceCount];
    float3 specularReflectionVectors[rayBounceCount];

    let reflectedLightRayOrigin = ray.origin + (earthNormal * shadowAcneCompensationValue);
    for(int i = 0; i < rayBounceCount; ++i) {
        let reflectedLightRayDir = normalize(earthNormal + getRandomUnitVector(threadId, 71 * i));
        let reflectedLightRay = Ray(reflectedLightRayOrigin, reflectedLightRayDir);

        reflectedRays[i] = reflectedLightRay;

        // Calculate atmospheric dimming
        let raycastResult = castRayAgainstOblateSpheroidFull(reflectedLightRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT);
        // if(!raycastResult.collidesSecond) {
        //     reflectedRayColor[i] = spaceColor;
        // }
        // else {
            scatterResult atmosray = calculateAtmosphericDimming(
                ray.origin,
                raycastResult.secondPosition,
                sunVector);
            reflectedRayColor[i] = atmosray.outscatter;
        //

        reflectedRayCosines[i] = dot(reflectedLightRayDir, earthNormal);

        let testVec = -sunVector;
        specularReflectionVectors[i] = normalize(2.0 * dot(earthNormal, testVec) * earthNormal - testVec);
    }

    return RayBounceResult(textureColor, materialSample, earthNormal, reflectedRays, reflectedRayColor, reflectedRayCosines, specularReflectionVectors);
}

float3 getReflectedRayColor(
        const float4 earthMaterial,
        const float3 earthNormal,
        const float3 reflectedLightColor,
        const float reflectedCosine,
        const float3 incomingRay,
        const float3 specularReflectionRay,
) {
    let baseSpecularCoeff = clamp(earthMaterial.x, 0.0, 1.0);

    let fresnelFactor =
        baseSpecularCoeff +
        ((1.0 - baseSpecularCoeff) *
          pow(1.0 - max(0.0, dot(earthNormal, normalize(-incomingRay))), 5.0));

    //let specularLightCoefficient = fresnelFactor * clamp(earthMaterial.x, 0.0, 1.0);
    //let diffuseLightCoefficient = 1.0 - specularLightCoefficient;

    let specularLightIntensity =
        fresnelFactor *
        pow(max(0.0, dot(normalize(-incomingRay), specularReflectionRay)), earthSpecularExponent);

    let diffuseLightIntensity = (1 - fresnelFactor) * max(0.0, reflectedCosine);

    let earthSurfaceLight = (specularLightIntensity) + (diffuseLightIntensity);

    return earthColorLightMultiplier * earthSurfaceLight * reflectedLightColor;
}

bool isRayHittingSun(const Ray ray) {
    return dot(ray.direction, -sunVector) > 0.0;
}

float3 getOblateSpheroidNormalVector(const float3 position, const float width, const float height) {
    return normalize(float3(position.xy / (width * width), position.z / (height * height)));
}

float3 getSpaceColor(const int2 threadId, const Ray cameraRay) {
    let raycastResult = castRayAgainstOblateSpheroidFull(cameraRay, ATMOSPHERE_WIDTH, ATMOSPHERE_HEIGHT);

    if(raycastResult.collidesFirst && raycastResult.collidesSecond) {
        // We hit the atmosphere
        return calculateAtmosphericDimming(
                   raycastResult.firstPosition,
                   raycastResult.secondPosition,
                   sunVector).inscatter;
    }

    // Very temporary star thingy
    if(pcgHashRandom((threadId.x) * (threadId.y) * 17) > 0.999) {
        return starColor;
    }
    else if(pcgHashRandom((threadId.x - 1) * (threadId.y) * 17) > 0.999) {
        return starColor;
    }
    else if(pcgHashRandom((threadId.x + 1) * (threadId.y) * 17) > 0.999) {
        return starColor;
    }
    else if(pcgHashRandom((threadId.x) * (threadId.y - 1) * 17) > 0.999) {
        return starColor;
    }
    else if(pcgHashRandom((threadId.x) * (threadId.y + 1) * 17) > 0.999) {
        return starColor;
    }

    return spaceColor;
}
