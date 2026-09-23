module constants;

// Earth Size
public static const float EARTH_WIDTH = 6378137;
public static const float EARTH_HEIGHT = 6356752;

public static const float widthOverHeight = EARTH_WIDTH / EARTH_HEIGHT;
public static const float earthWidthPow2 = EARTH_WIDTH * EARTH_WIDTH;

// Atmosphere
public static const float ATMOSPHERE_WIDTH = 6378137 + 11000*1;
public static const float ATMOSPHERE_HEIGHT = 6356752 + 11000*1;

// Other Constants
public static const float DEG_TO_RAD = 3.14159265358979 / 180.0;
public static const float RAD_TO_DEG = 180.0 / 3.14159265358979;

public static const float PI = 3.14159265358979323846264338327950288419716939937510582;

// Types
public struct Ray {
    public float3 origin;
    public float3 direction;
}

public struct SimpleRaycastResult {
    public bool collides;
    public float3 position;
}

public struct ComplexRaycastResult {
    public float3 firstPosition;
    public bool collidesFirst;

    public float3 secondPosition;
    public bool collidesSecond;
}

public struct scatterResult {
    public float3 outscatter;
    public float3 inscatter;
}