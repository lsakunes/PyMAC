module intersection;
import constants;

// Calculates both collision positions for the raycast
public ComplexRaycastResult castRayAgainstOblateSpheroidFull(const Ray ray, const float width, const float height) {
    let sphereSpaceRay = convertRayToSphereSpace(ray, width, height);

    // The pieces of the quadratic formula
    let a = dot(sphereSpaceRay.direction, sphereSpaceRay.direction);
    let b = 2.0 * dot(sphereSpaceRay.origin, sphereSpaceRay.direction);
    let c = dot(sphereSpaceRay.origin, sphereSpaceRay.origin) - (width * width);

    let determinant = ((-4.0 * c) * a) + (b * b);

    // Check if the ray misses the ellipsoid
    if (determinant < 0.0) {
        return ComplexRaycastResult(float3(9.0), false, float3(0.0), false);
    }

    // This will give us the smallest value of t, and this is almost always what we want,
    // but we need to throw this out in the case where t is negative. This would be the case
    // where the earth is behind the camera, and we don't want earth to be visible in that case.
    // We don't know if a is positive or negative, so we need to evaluate the whole thing to determine t's sign.
    let sqrtDeterminant = sqrt(determinant);
    let twoA = 2.0 * a;

    let smallT = (-b - sqrtDeterminant) / twoA;
    let largeT = (-b + sqrtDeterminant) / twoA;

    var smallTPosition = float3(0.0);
    var largeTPosition = float3(0.0);

    if (smallT >= 0.0) {
        // This is in sphere space still so we need to convert it back before returning it
        let sphereSpaceHitLocation = (sphereSpaceRay.direction * smallT) + sphereSpaceRay.origin;

        smallTPosition = convertPosToEllipsoidSpace(sphereSpaceHitLocation, width, height);
    }

    if (largeT >= 0.0) {
        // This is in sphere space still so we need to convert it back before returning it
        let sphereSpaceHitLocation = (sphereSpaceRay.direction * largeT) + sphereSpaceRay.origin;

        largeTPosition = convertPosToEllipsoidSpace(sphereSpaceHitLocation, width, height);
    }

    return ComplexRaycastResult(
        smallTPosition,
        smallT >= 0.0,
        largeTPosition,
        largeT >= 0.0);
}

public SimpleRaycastResult castRayAgainstOblateSpheroid(const Ray ray, const float width, const float height) {
    let raycastResult = castRayAgainstOblateSpheroidFull(ray, width, height);

    return SimpleRaycastResult(raycastResult.collidesFirst, raycastResult.firstPosition);
}

//////////////////////////
// Coordinate Conversions
//////////////////////////
public float3 convertPosToSphereSpace(const float3 pos) {
    return float3(pos.xy, pos.z * widthOverHeight);
}

public float3 convertPosToSphereSpace(const float3 pos, const float width, const float height) {
    return float3(pos.xy, pos.z * (width / height));
}

public float3 convertPosToEllipsoidSpace(const float3 pos) {
    return float3(pos.xy, pos.z / widthOverHeight);
}

public float3 convertPosToEllipsoidSpace(const float3 pos, const float width, const float height) {
    return float3(pos.xy, pos.z / (width / height));
}

public Ray convertRayToSphereSpace(const Ray ray) {
    return {
        float3(ray.origin.xy, ray.origin.z * widthOverHeight),
        normalize(float3(ray.direction.xy, ray.direction.z * widthOverHeight))
    };
}

public Ray convertRayToSphereSpace(const Ray ray, const float width, const float height) {
    let widthDivHeight = width / height;
    return {
        float3(ray.origin.xy, ray.origin.z * widthDivHeight),
        normalize(float3(ray.direction.xy, ray.direction.z * widthDivHeight))
    };
}

public Ray convertRayToEllipsoidSpace(const Ray ray) {
    return {
        float3(ray.origin.xy, ray.origin.z / widthOverHeight),
        normalize(float3(ray.direction.xy, ray.direction.z / widthOverHeight))
    };
}

public Ray convertRayToEllipsoidSpace(const Ray ray, const float width, const float height) {
    let widthDivHeight = width / height;
    return {
        float3(ray.origin.xy, ray.origin.z / widthDivHeight),
        normalize(float3(ray.direction.xy, ray.direction.z / widthDivHeight))
    };
}