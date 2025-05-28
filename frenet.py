import numpy as np

def get_frenet_coords(x, y, centerline, headings):
    '''Computes the Frenet coordinates (s,d) of a point in cartesian coordinates (x,y) w.r.t to the centerline'''    
    # Compute closest point on the centerline
    dx = centerline[:,0] - x
    dy = centerline[:,1] - y
    distances = np.hypot(dx,dy)
    idx = np.argmin(distances)

    # Compute arc length s
    s = np.sum(np.hypot(np.diff(centerline[:idx+1, 0]), np.diff(centerline[:idx+1, 1])))

    # Compute deviation d
    path_heading = headings[idx]
    normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
    rel_pos = np.array([x, y]) - centerline[idx]
    d = np.dot(rel_pos, normal)

    return s, d

def compute_curvature(centerline):
    '''Computes the curvature Kappa along the centerline'''
    x = centerline[:, 0]
    y = centerline[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    num = dx * ddy - dy * ddx
    denom_raw = (dx**2 + dy**2) ** 1.5
    # Avoid division by 0
    denom = np.where(np.abs(denom_raw) < 1e-6, 1e-6, denom_raw)
    kappa = np.divide(num, denom)
    return kappa