# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:36:23 2018

@author: simon
"""

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)    
    
def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def GetSkew(x):
    return np.asmatrix(
           [[0, -x[0,2], x[0,1]],
            [x[0,2], 0, -x[0,0]],
            [-x[0,1], x[0,0], 0]])

def vector_rotation(v1, v2):
    
    mat = np.asmatrix
    norm = np.linalg.norm 
    
    v1 = np.asmatrix(v1)
    v2 = np.asmatrix(v2)
    
    # rotation vector
    w = np.cross(v1,v2)
    if norm(w)!=0.:
        w = w / norm(w)
    else:
        w = np.asarray([0.,0.,0.])
    
    w_hat = GetSkew(w)
    
    #rotation angle
    cos_tht = v1.T*v2/(norm(v1)*norm(v2))
    tht = np.squeeze(mat(np.arccos(cos_tht)))
    sin_tht = np.sin(tht)
    w_hat2 = np.square(w_hat)
    tht1 = 1. - np.cos(tht)
    R = np.identity(3) + w_hat*sin_tht.T + (w_hat2*tht1.T)
    
    return R
    
def rotate_points(points,R):
    return np.dot(points,np.transpose(R))
    
def align_vector_rotation_2(A,B):
    #x = np.cross(A,B)
    #x = x / np.linalg.norm(x)
    
    An = A / np.linalg.norm(A)
    Bn = B / np.linalg.norm(B)
    #print(An,Bn,np.dot(An,Bn)/np.dot(An,Bn))
    x = np.cross(An,Bn)
    theta = np.arccos(np.dot(An,Bn))#/np.dot(An,Bn))
    #print('Theta',theta)

    M = [[0., -x[2], x[1]],
         [x[2], 0., -x[0]],
         [-x[1], x[0], 0.]]
    M2 = np.matmul(M,M)
    #print(M2)
    return np.eye(3) + np.dot(np.sin(theta),M) + np.dot((1.-np.cos(theta)),M2)
    
def align_vector_rotation(A,B):
    
    """ To align vector A to B:
    B = np.transpose(U*A.T))
    """
    
    mat = np.asmatrix
    norm = np.linalg.norm
    
    A = mat(A)/norm(A)
    B = mat(B)/norm(B)
    #print('A={}'.format(A))
    #print('B={}'.format(B))
    
    if np.all(A==B):
        #print('All equal')
        return np.identity(3)
    elif np.all(A==-B):
        #print('All negative')
        return -np.identity(3)
    
    G = np.matrix(
         [[A*B.T, mat(-norm(np.cross(A,B))), 0.],
         [norm(np.cross(A,B)), A*B.T,  0.],
         [0., 0., 1.],
         ])

    Fi = np.asarray([ A , (B - (A*B.T)*A) / norm((B - (A*B.T)*A)) , np.cross(B,A) ])
    Fi = np.matrix(np.squeeze(Fi))
    Fi = np.nan_to_num(Fi)
    try:
        U = Fi * G * np.linalg.inv(Fi)
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
    
    return U
    
#def rot_matrix(angle, direction, point=None):
#    """Return matrix to rotate about axis defined by point and direction.
#
#    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
#    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
#    True
#    >>> angle = (random.random() - 0.5) * (2*math.pi)
#    >>> direc = numpy.random.random(3) - 0.5
#    >>> point = numpy.random.random(3) - 0.5
#    >>> R0 = rotation_matrix(angle, direc, point)
#    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
#    >>> is_same_transform(R0, R1)
#    True
#    >>> R0 = rotation_matrix(angle, direc, point)
#    >>> R1 = rotation_matrix(-angle, -direc, point)
#    >>> is_same_transform(R0, R1)
#    True
#    >>> I = numpy.identity(4, numpy.float64)
#    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
#    True
#    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
#    ...                                               direc, point)))
#    True
#
#    """
#    sina = np.sin(angle)
#    cosa = np.cos(angle)
#    direction = unit_vector(direction[:3])
#    # rotation matrix around unit vector
#    R = np.diag([cosa, cosa, cosa])
#    R += np.outer(direction, direction) * (1.0 - cosa)
#    direction *= sina
#    R += np.array([[ 0.0,         -direction[2],  direction[1]],
#                      [ direction[2], 0.0,          -direction[0]],
#                      [-direction[1], direction[0],  0.0]])
#    M = np.identity(4)
#    M[:3, :3] = R
#    if point is not None:
#        # rotation not around origin
#        point = np.array(point[:3], dtype=np.float64, copy=False)
#        M[:3, 3] = point - np.dot(R, point)
#    return M
    
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return r, elev, az
    
def distance_point_line(x0,x1,x2):
    """
    Calculate the distance between a point x0 and a line
    defined by points x1 and x2
    """
    norm = np.linalg.norm
    return norm(np.cross(x0-x1,x0-x2)) / norm(x2-x1)
    
def distance_point_plane(p,p0,v):
    """
    Calculate the distance between a point p and a plane
    defined by point p0 and unit vector v pointing 'upwards'
    """
    a = -v[0]*p0[0] - v[1]*p0[1] - v[2]*p0[2]
    return np.abs(v[0]*p[0] + v[1]*p[1] + v[2]*p[2] + a) / np.linalg.norm(v)
    
def point_above_plane(p,v,p0):
    """
    Tests if a point p is on/above or below a plane, where
    p0 is a point on that plane and v is a unit vector pointing
    'upwards'.
    
    """
    if np.dot(v,p-p0)<0: # if==0, point is on plane
        return False
    else:
        return True
        
def point_inside_cylinder(p,x0,x1,r):
    """
    Tests if a point p is inside a cylinder, with central line defined by x0 and x1
    and radius r.
    """
    # Is the point outside the planes defined by the end circles?
    pap1 = point_above_plane(p,x1-x0,x1)
    pap0 = point_above_plane(p,x0-x1,x0)
    if pap1 or pap0:
        return False
    elif distance_point_line(p,x0,x1)>=r: # Is the point within the infinitely long cylinder?
        return False
    else:
        return True
        
def point_beyond_cylinder_ends(p,x0,x1,r):
    """
    Tests if a point p is beyond the ends of the ends of a cylinder with central line defined by x0 and x1
    and radius r.
    """
    pap1 = point_above_plane(p,x1-x0,x1)
    pap0 = point_above_plane(p,x0-x1,x0)
    if pap1 or pap0:
        return False
    else:
        return True
        
def sphere_inside_cylinder(p,sr,x0,x1,r):
    """
    Tests if a sphere (centre, p; radius, sr) overlaps with a cylinder, with central line defined by x0 and x1
    and radius r.
    """
    norm = np.linalg.norm
    
    # Is the sphere outside the planes defined by the end circles?
    l = norm(x1-x0)
    pap1 = point_above_plane(p,x1-x0,x1)
    d1 = distance_point_plane(p,x1,(x1-x0)/l)
    if pap1 and d1>sr:
        return False
    pap0 = point_above_plane(p,x0-x1,x0)
    d0 = distance_point_plane(p,x0,(x0-x1)/l)
    if pap0 and d0>sr:
        return False
    if distance_point_line(p,x0,x1)>=r+sr: # Is the sphere within the infinitely long cylinder?
        return False
    else:
        return True
    
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])