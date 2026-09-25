#region imports
from atmos import (
    calculateAtmosphericDimming,
    earth_hit_km,
    atmosphere_segment_m,
    luminance,
    RGB,
)
import numpy as np
import cv2 as cv
import random
#import sympy as sp
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool, shared_memory
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
num_cores = os.cpu_count()
#endregion





#region PARAMETERS

# !! WORLD AND CAMERA COORDS HAVE Z FORWARD Y UP


#### SUN/EARTH ##############################
sunAngle = 0.4
sun = np.array([0,np.sin(sunAngle),np.cos(sunAngle)])

axes = [6378.137, 6378.137, 6356.752]  # WGS84 equatorial / polar, km
atmosphere = 11  # km shell thickness (matches old ATMOSPHERE_* offset)

## includes atmosphere cause that what edge detection thinks is the horizon
diagAxes = np.array([ [(axes[0])+atmosphere,              0,             0  ],
                      [      0,          (axes[1])+atmosphere,           0  ],
                      [      0,                0,       (axes[2])+atmosphere]])

diagInvAxes = np.array([ [1/(axes[0]+atmosphere),         0,               0      ],
                      [      0,          1/(axes[1]+atmosphere),        0      ],
                      [      0,                 0,       1/(axes[2]+atmosphere)]])

## for when we apply MAC and start working with the actual horizon
diagTrueAxes = np.array([ [(axes[0]),              0,             0  ],
                      [      0,          (axes[1]),           0  ],
                      [      0,                0,       (axes[2])]])

diagTrueInvAxes = np.array([ [1/(axes[0]),         0,               0      ],
                      [      0,          1/(axes[1]),        0      ],
                      [      0,                 0,       1/(axes[2])]])

Ap = diagTrueInvAxes*diagTrueInvAxes
#############################################

### CAMERA INTRINSICS #######################
focalLength     = 50 * 0.001 # 50mm to m
sensorWidth     = 36 * 0.001 # 36mm to m
sensorHeight    = 36 * 0.001 # 36mm to m
xResolution     = 1024
yResolution     = 1024

xPitch = sensorWidth/xResolution
yPitch = sensorWidth/yResolution
dx = focalLength/xPitch
dy = focalLength/yPitch
#############################################

### CAMERA POSITION/ORIENTATION #############
thetay = -80
thetax = 0
TPCY = np.array([[np.cos(thetay),  0,  np.sin(thetay)],
                [     0,          1,      0        ],
                [-np.sin(thetay),  0,  np.cos(thetay)]])

TPCX = np.array([[1,  0,  0],
                [0, np.cos(thetax), -np.sin(thetax)],
                [0, np.sin(thetax),  np.cos(thetax)]])
TPC = TPCX.dot(TPCY)
TCP = np.linalg.matrix_transpose(TPC)


KMat = np.array([[dx, 0, xResolution/2],
                 [0, dy, yResolution/2],
                 [0,0,1]])
invKMat = np.array([[1/dx, 0, -xResolution/(2*dx)],
                    [0, 1/dy, -yResolution/(2*dy)],
                    [0,0,1]])


## position in world coords
rp = np.array([-30000,0,0])
## position in camera coords
rc = TPC.dot(rp)
## sun vector in camera coords
sunc = TPC.dot(sun)
## sun in image coords
sunimgBad = np.array([sunc[0], sunc[1]])
#normalized
sunImg = sunimgBad/np.linalg.norm(sunimgBad)

Ac = TPC.dot(Ap.dot(TCP))

## the conic section we see 
C = ((np.outer(Ac.dot(rc),(Ac.dot(rc))) - (rc.dot(Ac.dot(rc)) * np.eye(3) - np.eye(3)).dot(Ac))*10**8)*-1
#############################################

#endregion




#region HELPER FUNCTIONS
def minPosQuadratic(a, b, c):
    discriminant = (b*b - 4*a*c)
    if (discriminant<0):
        # put in funky numbers for debugging
        return -88888
    x1 = (-b + np.sqrt(discriminant))/(2*a)
    x2 = (-b - np.sqrt(discriminant))/(2*a)
    if (x1 < 0):
        if (x2 < 0):
            return -99999
        return x2
    if (x2 < 0):
        return x1
    return np.min([x1, x2])

# returns the root with the smallest magnitude
def minAbsQuadratic(a, b, c):
    discriminant = (b*b - 4*a*c)
    if (discriminant<0):
        return -99999
    x1 = (-b + np.sqrt(discriminant))/(2*a)
    x2 = (-b - np.sqrt(discriminant))/(2*a)
    return x1 if (abs(x1) < abs(x2)) else x2

# world to image coords
def K(x, y):
    u = dx*x + xResolution/2
    v = dy*y + yResolution/2
    return (u,v)

def Knp(x):
    u = dx*x[0] + xResolution/2
    v = dy*x[1] + yResolution/2
    return np.array([u,v])

def KInv(u, v):
    x = u/dx - xResolution/(2*dx)
    y = v/dy - yResolution/(2*dy)
    return x,y

def insideEarth(x, y, _C, _det):
    xBar = np.array([x,y,1])
    if (xBar.dot(rc) < 0):
        return 0
    if (xBar.dot(_C.dot(xBar))*-np.sign(_det) < 0):
        return 1
    return 0

# camera position in world coords earth center
def earth(pos, dir, sun):
    hit = earth_hit_km(pos, dir)
    if hit is None:
        return 0

    surface_km = hit
    surface_dir = surface_km / np.linalg.norm(surface_km)
    ndotl = surface_dir.dot(sun)
    if ndotl < 0:
        return 0
    return ndotl

def atmos(pos, dir, sun):
    """Returns the atmosphere's RGB contribution along this ray (an additive sky/dimming color)."""

    segment = atmosphere_segment_m(pos, dir, sun)
    if segment is None:
        return RGB.zeros()

    start_m, end_m, path_kind = segment
    result = calculateAtmosphericDimming(start_m, end_m, sun)

    if path_kind == "limb":
        return result.inscatter

    surface_ndotl = max(
        0.0,
        np.dot(end_m / np.linalg.norm(end_m), sun / np.linalg.norm(sun)),
    )
    gray = 1.0
    combined = result.inscatter + result.outscatter * gray * surface_ndotl
    return combined - RGB.ones() * surface_ndotl

def rgb_to_channels(color: RGB) -> np.ndarray:
    return np.array([color.r, color.g, color.b], dtype=np.float64)

# in pixel coords
def rayConicIntersection(c, pixel, dir):
    nx = dir[0]
    ny = dir[1]
    x = pixel[0]
    y = pixel[1]
    a = nx*nx*c[0][0] + 2*nx*ny*c[1][0] + ny*ny*c[1][1]
    b = nx*x*2*c[0][0] + (y*nx + x*ny)*2*c[1][0] + ny*y*2*c[1][1] + nx*2*c[0][2] + ny*2*c[1][2]
    c = np.array([x,y,1]).dot(c.dot(np.array([x,y,1])))
    return minPosQuadratic(a,b,c)

def midPixColor(img, x, y):
    start = img[(int)(y), (int)(x)]
    corner = img[(int)(y)+1, (int)(x)+1]
    xPixel = img[(int)(y), (int)(x)+1]
    yPixel = img[(int)(y)+1, (int)(x)]
    x = x-(int)(x)
    y = y-(int)(y)
    return (y)*(x*corner+(1-x)*yPixel)+(1-y)*(x*xPixel+(1-x)*start)

def subpixelDiff(dir, origin, colorIn, img):
    horizontalSign = np.sign(dir[0])
    verticalSign = np.sign(dir[1])
    renorm = 1
    color = np.int32(colorIn)
    start = np.int32(midPixColor(img, origin[0], origin[1]))
    corner = np.int32(midPixColor(img, origin[0]+horizontalSign, origin[1]+verticalSign))
    yPixel = np.int32(midPixColor(img, origin[0], origin[1]+verticalSign))
    xPixel = np.int32(midPixColor(img, origin[0]+horizontalSign, origin[1]))
    if (start == color):
        return 0
    
    # normalize such that the largest direction of movement is independent and considered to be x
    y = 1
    if ((abs(dir[0]) > abs(dir[1]))):
        y = abs(dir[1])/abs(dir[0])
        renorm = abs(dir[0])
    else:
        y = abs(dir[0])/abs(dir[1])
        renorm = abs(dir[1])
        xPixel, yPixel = yPixel, xPixel
    
    a = y*corner - y*yPixel - y*xPixel + y*start
    b = y*yPixel + xPixel - (y+1)*start
    c = start - color

    if (abs(a) < 0.01):
        if ((xPixel-start) == 0):
            return 0
        # basically linear
        return (color-start)/(xPixel-start)
    t = minAbsQuadratic(a,b,c)
    if (t == -99999): 
        print(f"\n\ncolor: {color}")
        print(f"start: {start}")
        print(f"xPixel: {xPixel}")
        print(f"yPixel: {yPixel}")
        print(f"corner: {corner}")
        print(f"dir: {dir}")
        raise Exception("dumbass")
    return t/renorm

#endregion


def thingy(shm_name, shape, dtype, shm_color_name, color_shape, color_dtype, offset):
    shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    shmColor = shared_memory.SharedMemory(name=shm_color_name)
    aColor = np.ndarray(color_shape, dtype=color_dtype, buffer=shmColor.buf)
    i = 0
    j = offset
    while i < xResolution:
        while j < yResolution:
            x, y = KInv(i, j)
            vec = np.array([x,y,1])
            vec = vec/np.linalg.norm(vec)
            brightness = earth(rp, vec, sunc)
            a[j][i] = brightness * 255
            atmosColor = atmos(rc, vec, -sunc)
            atmo = luminance(atmosColor) * 255
            a[j][i] = max(atmo+a[i][j], 0)
            colorPixel = np.clip((brightness * 255) + rgb_to_channels(atmosColor) * 255, 0, 255)
            aColor[j][i] = colorPixel
            if (i%10 == 0 and j%255 == 0):
                print(f"{i},{j} from thread {offset}")
            if (a[j][i] > 255):
                a[j][i] = 255
            j += num_cores
        j = offset
        i += 1
    shm.close()
    shmColor.close()

#region GRAPHICS
def box(pt, pts, color, size):
    for i in range(size*2):
        for j in range(size*2):
            if (pt[0]+i-size, pt[1]+j-size) in pts:
                return 0
    return color  

if __name__ == "__main__":
    #region GRAPHICS
    shm = shared_memory.SharedMemory(create=True, size=xResolution*yResolution)
    a = np.ndarray((xResolution, yResolution), dtype=np.uint8, buffer=shm.buf)
    a[:] = 0
    shmColor = shared_memory.SharedMemory(create=True, size=xResolution*yResolution*3)
    aColor = np.ndarray((xResolution, yResolution, 3), dtype=np.uint8, buffer=shmColor.buf)
    aColor[:] = 0
    with Pool(num_cores) as p:
            p.map(partial(thingy, shm.name, a.shape, a.dtype, shmColor.name, aColor.shape, aColor.dtype), range(num_cores))
    a = a.copy()
    aColor = aColor.copy()
    shm.close()
    shm.unlink()
    shmColor.close()
    shmColor.unlink()

    a=cv.GaussianBlur(a, (3, 3), 0)
    aColor=cv.GaussianBlur(aColor, (3, 3), 0)
    img = Image.fromarray(aColor)
    img.save("./output.png")
    #endregion




    #region EDGE
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(a, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(a, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    ## USED FOR MAC
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    gradDraw = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    corner = (0.5+0.5*np.sign(sunImg[1]), 0.5+0.5*np.sign(sunImg[0]))
    dist = np.sqrt(yResolution**2+xResolution**2)
    scan = -yResolution
    orth = np.sign(corner[1]-0.5)*np.array([-sunImg[1], sunImg[0]])

    pts = []
    while (scan < xResolution):
        start = corner + orth*scan
        stepper = 0
        while(stepper < dist):
            pt = start + sunImg*stepper
            stepper += 1
            if (pt[0] > xResolution or pt[1] > yResolution):
                break
            if (pt[0] < 0 or pt[1] < 0):
                continue
            if (grad[(int)(pt[1])][(int)(pt[0])] > 30):
                color = 30
                if ((np.array([xResolution/2, yResolution/2]-pt)/np.linalg.norm(np.array([xResolution/2, yResolution/2]-pt))).dot(sunImg) > 0.342):
                    pts.append([pt[0], pt[1]])
                break
        scan += 1
    img = Image.fromarray(cv.cvtColor(gradDraw, cv.COLOR_GRAY2RGB))
    img.save("./grad.jpg")

    img = Image.fromarray(aColor)
    img.save("./outputEdge.png")
    #endregion



    #region CRA
    imageToSpace = diagInvAxes.dot(TPCY.transpose()).dot(invKMat)
    pointsSize = len(pts)
    normalizedVecsToHorizonMat = np.zeros((pointsSize, 3), dtype=np.float32)
    for i in range(pointsSize):
        pBar = np.array([pts[i][0], pts[i][1], 1])
        vecToHorizon = (imageToSpace.dot(pBar))
        normalizedVecToHorizon = vecToHorizon/np.linalg.norm(vecToHorizon)
        for j in range(3):
            normalizedVecsToHorizonMat[i, j] = normalizedVecToHorizon[j]
    vecToEarth = np.linalg.lstsq(normalizedVecsToHorizonMat, np.ones(pointsSize, dtype=np.float32), rcond=None)[0]
    vecToEarth = (TPC.dot(diagAxes).dot(vecToEarth)) * (1/np.sqrt(vecToEarth.dot(vecToEarth) - 1))
    print(f"error: {rc-vecToEarth}")
    print(f"error: {np.linalg.norm(rc)-np.linalg.norm(vecToEarth)}")
    sBar = KMat.dot(vecToEarth/vecToEarth[2])

    C = ((np.outer(Ac.dot(vecToEarth),(Ac.dot(vecToEarth))) - (vecToEarth.dot(Ac.dot(vecToEarth)) * np.eye(3) - np.eye(3)).dot(Ac)))
    C = C/C[0][0]
    Cuv = np.linalg.matrix_transpose(invKMat).dot(C.dot(invKMat))
    Cdet = np.linalg.det(C)
    #endregion





    #region ATMOS
    b = np.zeros((xResolution, yResolution), dtype=np.uint8)
    bColor = np.zeros((xResolution, yResolution, 3), dtype=np.uint8)
    ptBrightness = []

    for i in range(xResolution):
        for j in range(yResolution):
            x, y = KInv(i, j)
            vec = np.array([x,y,1])
            vec = vec/np.linalg.norm(vec)
            dist = np.linalg.norm(rc)
            brightness, surface = earth(vecToEarth, vec, sunc)
            b[j][i] = brightness * 255
            atmosColor = atmos(vecToEarth, vec, sunc)
            b[j][i] += (luminance(atmosColor))*255
            bColor[j][i] = np.clip((brightness * 255) + rgb_to_channels(atmosColor) * 255, 0, 255)
            if (b[j][i] > 255):
                b[j][i] = 255
    b=cv.GaussianBlur(b, (3, 3), 0)
    bColor=cv.GaussianBlur(bColor, (3, 3), 0)
    modelVisual = Image.fromarray(bColor)
    modelVisual.save("./model.png")

    kernel1d = cv.getGaussianKernel(3, 0)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    for pt in pts:
    
        brightness = 0
        for k in range(3):
            for l in range(3):
                x, y = KInv(pt[0]+k-1, pt[1]+l-1)
                vec = np.array([x,y,1])
                vec = vec/np.linalg.norm(vec)
                dist = np.linalg.norm(rc)
                dummy, surface = earth(vecToEarth, vec, sunc)
                brightness += (luminance(atmos(vecToEarth, vec, sunc))*255) * kernel2d[k][l]
        if (brightness > 255):
                brightness = 255
        ptBrightness.append(brightness)
        brightness = 0
        colorBrightness = np.zeros(3)
        for k in range(3):
            for l in range(3):
                x, y = KInv((int)(pt[0])+k-1, (int)(pt[1])+l-1)
                vec = np.array([x,y,1])
                vec = vec/np.linalg.norm(vec)
                dist = np.linalg.norm(rc)
                dummy, surface = earth(vecToEarth, vec, sunc)
                atmosColor = atmos(vecToEarth, vec, sunc)
                brightness += (luminance(atmosColor)*255) * kernel2d[k][l]
                colorBrightness += (rgb_to_channels(atmosColor)*255) * kernel2d[k][l]
        b[(int)(pt[1]), (int)(pt[0])] = brightness
        bColor[(int)(pt[1]), (int)(pt[0])] = np.clip(colorBrightness, 0, 255)

    modelVisual = Image.fromarray(bColor)
    modelVisual.save("./modelEdge.png")
    c = np.zeros((xResolution, yResolution), dtype=np.uint8)
    for i in range(xResolution):
        for j in range(yResolution):
            c[j][i] = (255+a[i][j]-b[i][j])
    diff = Image.fromarray(cv.cvtColor(c, cv.COLOR_GRAY2RGB))
    diff.save("./diff.png")
    #endregion




    #region MATCH
    normOffsets = np.zeros((len(pts), 2), dtype=np.float32)
    lambdas = np.zeros((len(pts)), dtype=np.float32)
    moveSum = 0
    i = 0
    for pt in pts:
        x, y = pt[0], pt[1]
        # point to earth center
        offset = np.array([sBar[0], sBar[1]])-pt
        normOffset = offset/np.linalg.norm(offset)
        normOffsets[i] = normOffset
        lambd = rayConicIntersection(Cuv, pt, normOffset)
        lambdas[i] = lambd
        diff = subpixelDiff(normOffset, (x, y), ptBrightness[i], a) #b[(int)(y)][(int)(x)]
        moveSum += diff
        i += 1

    img = Image.fromarray(a)
    img.save("./outputAWPOFEIJ.png")
    move = moveSum/i
    i = 0
    step = 0.5
    print(f"adjusting by {move*step}")
    for pt in pts:
        x, y = pt[0], pt[1]
        lambd = lambdas[i]
        lamPixOffset = np.array([lambd*normOffsets[i][0], lambd*normOffsets[i][1]])
        pts[i] = pts[i] + lamPixOffset + normOffsets[i]*step*move
        i += 1


    img = Image.fromarray(a)
    img.save("./outputAWPOFEIJ.png")
    #endregion





    #region RERUN CRA
    imageToSpace = diagTrueInvAxes.dot(TPCY.transpose()).dot(invKMat)
    pointsSize = len(pts)
    normalizedVecsToHorizonMat = np.zeros((pointsSize, 3), dtype=np.float32)
    for i in range(pointsSize):
        pBar = np.array([pts[i][0], pts[i][1], 1])
        vecToHorizon = (imageToSpace.dot(pBar))
        normalizedVecToHorizon = vecToHorizon/np.linalg.norm(vecToHorizon)
        for j in range(3):
            normalizedVecsToHorizonMat[i, j] = normalizedVecToHorizon[j]
    vecToEarth = np.linalg.lstsq(normalizedVecsToHorizonMat, np.ones(pointsSize, dtype=np.float32), rcond=None)[0]
    vecToEarth = (TPC.dot(diagTrueAxes).dot(vecToEarth)) * (1/np.sqrt(vecToEarth.dot(vecToEarth) - 1))
    print(f"MAC adjusted error: {rc-vecToEarth}")
    print(f"MAC adjusted error: {np.linalg.norm(rc)-np.linalg.norm(vecToEarth)}")


    #endregion

    print("ran test 36")
