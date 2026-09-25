from atmos import (
    calculateAtmosphericDimming,
    earth_hit_km,
    atmosphere_segment_m,
    luminance,
    SIGMA,
    S_R,
    reload_scatter_profile,
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
from datetime import datetime
from pathlib import Path
from nrlmsise00 import msise_flat
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
num_cores = os.cpu_count()

# WORLD AND CAMERA COORDS HAVE Z FORWARD Y UP

#region PARAMETERS
#EARTH
axes = [6378.137, 6378.137, 6356.752]  # WGS84 equatorial / polar, km

#CAMERA
focalLength     = 50 * 0.001 # 50mm to m
sensorWidth     = 36 * 0.001 # 36mm to m
sensorHeight    = 36 * 0.001 # 36mm to m
xResolution     = 1024
yResolution     = 1024

xPitch = sensorWidth/xResolution
yPitch = sensorWidth/yResolution
dx = focalLength/xPitch
dy = focalLength/yPitch


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
artemisOffset = 95

diagAxes = np.array([ [(axes[0])+artemisOffset,              0,             0  ],
                      [      0,          (axes[1])+artemisOffset,           0  ],
                      [      0,                0,       (axes[2])+artemisOffset]])

diagInvAxes = np.array([ [1/(axes[0]+artemisOffset),         0,               0      ],
                      [      0,          1/(axes[1]+artemisOffset),        0      ],
                      [      0,                 0,       1/(axes[2]+artemisOffset)]])

diagTrueAxes = np.array([ [(axes[0]),              0,             0  ],
                      [      0,          (axes[1]),           0  ],
                      [      0,                0,       (axes[2])]])

diagTrueInvAxes = np.array([ [1/(axes[0]),         0,               0      ],
                      [      0,          1/(axes[1]),        0      ],
                      [      0,                 0,       1/(axes[2])]])

Ap = np.array([ [(1/axes[0])**2,         0,               0     ],
                [        0,      (1/axes[1])**2,          0     ],
                [        0,              0,       (1/axes[2])**2]])
Ac = TPC.dot(Ap.dot(TCP))

KMat = np.array([[dx, 0, xResolution/2],
                 [0, dy, yResolution/2],
                 [0,0,1]])
invKMat = np.array([[1/dx, 0, -xResolution/(2*dx)],
                    [0, 1/dy, -yResolution/(2*dy)],
                    [0,0,1]])


# just move backwards
rp = np.array([-30000,0,0])
rc = TPC.dot(rp)

rpEstimate = -1*np.array([0,0,-axes[0]-158])
rcEstimate = TPC.dot(rpEstimate)


C = ((np.outer(Ac.dot(rc),(Ac.dot(rc))) - (rc.dot(Ac.dot(rc)) * np.eye(3) - np.eye(3)).dot(Ac))*10**8)*-1

sunAngle = 0.4
sun = np.array([0,np.sin(sunAngle),np.cos(sunAngle)])
sunc = TPC.dot(sun)
sunimgBad = np.array([sunc[0], sunc[1]])
sunImg = sunimgBad/np.linalg.norm(sunimgBad)

atmosphere = 11  # km shell thickness; matches atmos.py ATMOSPHERE_* (+11 km) and MSIS 0–11 km table
#endregion

# NRLMSISE-00 N2/O2/Ar weighted Rayleigh table (built once at MAC startup)
_W_N2 = 1.00
_W_O2 = 0.91
_W_AR = 0.858
_ALT_MAX_KM = 11.0
_ALT_STEP_KM = 0.25
_MSIS_DATETIME = datetime(2009, 6, 21, 12, 0, 0)
_LAT_DEG = 45.0
_LON_DEG = 0.0
_F107A = 150.0
_F107 = 150.0
_AP = 4.0
_SCATTER_PROFILE_PATH = Path(__file__).resolve().parent / "scatter_profile.npz"


def build_scatter_profile():
    """Build MSIS-weighted beta_rgb table and write scatter_profile.npz."""
    alt_km = np.arange(0.0, _ALT_MAX_KM + 0.5 * _ALT_STEP_KM, _ALT_STEP_KM, dtype=np.float64)
    raw = np.asarray(
        msise_flat(_MSIS_DATETIME, alt_km.tolist(), _LAT_DEG, _LON_DEG, _F107A, _F107, _AP),
        dtype=np.float64,
    )
    n2 = raw[:, 2]
    o2 = raw[:, 3]
    ar = raw[:, 4]

    n_w = n2 * _W_N2 + o2 * _W_O2 + ar * _W_AR
    n_w0 = n_w[0]
    if n_w0 <= 0.0:
        raise RuntimeError("Sea-level weighted density N_w0 is non-positive; check MSIS output.")

    beta_rgb = (S_R * SIGMA)[None, :] * (n_w / n_w0)[:, None]

    np.savez(
        _SCATTER_PROFILE_PATH,
        alt_km=alt_km,
        beta_rgb=beta_rgb,
        n_N2=n2,
        n_O2=o2,
        n_Ar=ar,
        n_w=n_w,
        N_w0=np.array(n_w0),
        weights=np.array([_W_N2, _W_O2, _W_AR]),
        species=np.array(["N2", "O2", "Ar"]),
    )
    reload_scatter_profile()

    def nearest(h):
        return int(np.argmin(np.abs(alt_km - h)))

    i0, i5, i11 = nearest(0.0), nearest(5.0), nearest(11.0)
    print(f"Wrote {_SCATTER_PROFILE_PATH}")
    print(f"alts: {alt_km[0]:.2f} .. {alt_km[-1]:.2f} km  (n={len(alt_km)})")
    print(f"N_w0 (sea-level weighted dens): {n_w0:.6e} cm^-3")
    for label, i in (("0 km", i0), ("5 km", i5), ("11 km", i11)):
        print(
            f"  {label}: N2={n2[i]:.3e}  O2={o2[i]:.3e}  Ar={ar[i]:.3e}  "
            f"n_w/N_w0={n_w[i]/n_w0:.4f}  beta={beta_rgb[i]}"
        )


#region HELPER FUNCTIONS
def minPosQuadratic(a, b, c):
    discriminant = (b*b - 4*a*c)
    if (discriminant<0):
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

def minAbsQuadratic(a, b, c):
    discriminant = (b*b - 4*a*c)
    if (discriminant<0):
        return -99999
    x1 = (-b + np.sqrt(discriminant))/(2*a)
    x2 = (-b - np.sqrt(discriminant))/(2*a)
    return x1 if (abs(x1) < abs(x2)) else x2

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
        return 100
    return 0

def earth(radius, pos, dir, sun):
    del radius  # geometry comes from atmos.py WGS84 constants
    hit = earth_hit_km(pos, dir)
    if hit is None:
        return 0, 0

    surface_km, distance_km = hit
    surface_dir = surface_km / np.linalg.norm(surface_km)
    ndotl = surface_dir.dot(sun)
    if ndotl < 0:
        return 0, distance_km
    return ndotl, distance_km

def atmos(radius, pos, dir, sun, surface):
    del radius, surface  # geometry comes from atmos.py; surface kept for call-site compatibility

    segment = atmosphere_segment_m(pos, dir, sun)
    if segment is None:
        return 0.0

    start_m, end_m, path_kind = segment

    # Limb-only rays (miss earth): MAC N·sun lights the left/day side, but atmos.py
    # casts sun rays along -sunDirection. Pass -sun so the glow sits on the lit limb.
    if path_kind == "limb":
        result = calculateAtmosphericDimming(start_m, end_m, -sun)
        return luminance(result.inscatter)

    result = calculateAtmosphericDimming(start_m, end_m, sun)

    surface_ndotl = max(
        0.0,
        np.dot(end_m / np.linalg.norm(end_m), sun / np.linalg.norm(sun)),
    )
    # Night disk: no airglow wash; limb path above still contributes
    if surface_ndotl <= 0.0:
        return 0.0

    gray = 1.0
    combined = result.inscatter + result.outscatter * gray * surface_ndotl
    return float(luminance(combined) - surface_ndotl)

# def atmos(radius, pos, dir, sun, surface):
#     dot = dir.dot(pos)
#     det = (2*dot)**2 - 4*(pos.dot(pos)-radius*radius)
#     if (det < 0):
#         return 0
    
#     d1 = (dot + np.sqrt(det)/2)
#     d2 = (dot - np.sqrt(det)/2)
    
#     if (d1<0 or d2<0):
#         return 0

#     normPos = (pos/np.linalg.norm(pos))
#     atmosDir = (dir - dir.dot(normPos)*normPos)
#     atmosDir = atmosDir/(np.linalg.norm(atmosDir))
#     dot = -atmosDir.dot(sun)
#     if (dot>1):
#         print(dot)
#     if (dot < 0):
#         return 0
#     if (surface > 0):
#         if (abs(d1) > abs(d2)):
#             d1 = d2
#         return (abs(d1-surface)/(np.sqrt(radius**2-Ac[0,0]**2)/2)) * dot
#     return (abs(d1-d2)/(np.sqrt(radius**2-Ac[0,0]**2)/2)) * dot

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

    #color = np.int32(model[origin[1], origin[0]])
    color = np.int32(colorIn)
    start = np.int32(midPixColor(img, origin[0], origin[1]))
    corner = np.int32(midPixColor(img, origin[0]+horizontalSign, origin[1]+verticalSign))
    yPixel = np.int32(midPixColor(img, origin[0], origin[1]+verticalSign))
    xPixel = np.int32(midPixColor(img, origin[0]+horizontalSign, origin[1]))

    # print(f"\n\n{(origin[1], origin[0])} : {start}")
    # print(f"{(origin[1]+verticalSign, origin[0]+horizontalSign)} : {corner}")
    # print(f"{(origin[1], origin[0]+horizontalSign)} : {xPixel}")
    # print(f"{(origin[1]+verticalSign, origin[0])} : {yPixel}")

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
    # print(f"\na,b,c: {a},{b},{c} -> {color}")
    t = minAbsQuadratic(a,b,c)
    if (t == -99999): 
        print(f"\n\ncolor: {color}")
        print(f"start: {start}")
        print(f"xPixel: {xPixel}")
        print(f"yPixel: {yPixel}")
        print(f"corner: {corner}")
        print(f"dir: {dir}")
        raise Exception("dumbass")
    # print(f"\nrenorm:\t{renorm}")
    # print(f"t:\t{t}")
    # print(f"result:\t{t/renorm}\n")
    return t/renorm

#endregion


def thingy(shm_name, shape, dtype, offset):
    shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    i = 0
    j = offset
    while i < xResolution:
        while j < yResolution:
            x, y = KInv(i, j)
            vec = np.array([x,y,1])
            vec = vec/np.linalg.norm(vec)
            dist = np.linalg.norm(rc)
            brightness, surface = earth(axes[0], rc, vec, sunc)
            a[j][i] = brightness * 100
            a[j][i] += (atmos(axes[0]+atmosphere, rc, vec, sunc, surface))*100
            print(f"{i},{j} from thread {offset}")
            if (a[j][i] > 100):
                a[j][i] = 100
            j += num_cores
        j = offset
        i += 1
    shm.close()

#region GRAPHICS
def box(pt, pts, color, size):
    for i in range(size*2):
        for j in range(size*2):
            if (pt[0]+i-size, pt[1]+j-size) in pts:
                return 0
    return color  

if __name__ == "__main__":
    build_scatter_profile()

    #region GRAPHICS
    shm = shared_memory.SharedMemory(create=True, size=xResolution*yResolution)
    a = np.ndarray((xResolution, yResolution), dtype=np.uint8, buffer=shm.buf)
    a[:] = 0
    with Pool(num_cores) as p:
            p.map(partial(thingy, shm.name, a.shape, a.dtype), range(num_cores))
    a = a.copy()
    shm.close()
    shm.unlink()

            #a[i][j] = box((i,j), pts, a[i][j], 3)
            # if (i % 20 == 0 or j % 20 == 0):
            #     a[i][j] = 100 - a[i][j]
    a=cv.GaussianBlur(a, (3, 3), 0)
    img = Image.fromarray(a)
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
    img = Image.fromarray(gradDraw)
    img.save("./grad.jpg")

    img = Image.fromarray(a)
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
    ptBrightness = []

    for i in range(xResolution):
        for j in range(yResolution):
            x, y = KInv(i, j)
            vec = np.array([x,y,1])
            vec = vec/np.linalg.norm(vec)
            dist = np.linalg.norm(rc)
            brightness, surface = earth(axes[0], vecToEarth, vec, sunc)
            b[j][i] = brightness * 100
            b[j][i] += (atmos(axes[0]+atmosphere, vecToEarth, vec, sunc, surface))*100
            if (b[j][i] > 100):
                b[j][i] = 100
    b=cv.GaussianBlur(b, (3, 3), 0)
    modelVisual = Image.fromarray(b)
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
                dummy, surface = earth(axes[0], vecToEarth, vec, sunc)
                brightness += ((atmos(axes[0]+atmosphere, vecToEarth, vec, sunc, surface))*100) * kernel2d[k][l]
        if (brightness > 100):
                brightness = 100
        ptBrightness.append(brightness)
        brightness = 0
        for k in range(3):
            for l in range(3):
                x, y = KInv((int)(pt[0])+k-1, (int)(pt[1])+l-1)
                vec = np.array([x,y,1])
                vec = vec/np.linalg.norm(vec)
                dist = np.linalg.norm(rc)
                dummy, surface = earth(axes[0], vecToEarth, vec, sunc)
                brightness += ((atmos(axes[0]+atmosphere, vecToEarth, vec, sunc, surface))*100) * kernel2d[k][l]
        b[(int)(pt[1]), (int)(pt[0])] = brightness

    modelVisual = Image.fromarray(b)
    modelVisual.save("./modelEdge.png")
    c = np.zeros((xResolution, yResolution), dtype=np.uint8)
    for i in range(xResolution):
        for j in range(yResolution):
            c[j][i] = (100+a[i][j]-b[i][j])
    diff = Image.fromarray(c)
    diff.save("./diff.png")
    #endregion




    #region MATCH


    #ptOffset = np.zeros((len(pts), 2), dtype=np.uint8)
    #ptOffsetGrad = np.zeros((len(pts), 2), dtype=np.uint8)
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
        #ptOffset[i] = offset
        # moveSum += grad_x[(int)(y)][(int)(x)] * normOffset[0] + grad_y[(int)(y)][(int)(x)] * normOffset[1]
        # print(f"{(int)(pt[0])},{(int)(pt[1])} diff {(int)(pt[0])-pt[0]},{(int)(pt[1])-pt[1]}: {ptBrightness[i] - b[(int)(y)][(int)(x)]}")
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
