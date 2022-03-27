import math # atan, sin, cos
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cube_len = 2
focal_len = 0.4
pixel_size = [0.51, 0.49]
camera_position = np.array([0, 0, -0.25])
camera_orientation = np.array([0, 0.1, 0])

point_pos = cube_len / 2

vertices = [[point_pos, point_pos, point_pos], [point_pos, point_pos, -point_pos], [point_pos, -point_pos, -point_pos], [point_pos, -point_pos, point_pos], [-point_pos, point_pos, point_pos], [-point_pos, point_pos, -point_pos], [-point_pos, -point_pos, -point_pos], [-point_pos, -point_pos, point_pos]]

edges = [[0, 1], [0, 3], [0, 4], [1, 2], [2, 3], [2, 6], [1, 5], [4, 5], [5, 6], [3, 7], [4, 7], [6, 7]]

# these functions require non-homogenous coordinates
def show_3d(vertices, edges):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for edge in edges:
        p1 = vertices[edge[0]]
        p2 = vertices[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def show_2d(vertices, edges, size):
    fig = plt.figure()
    ax = fig.gca()
    for edge in edges:
        p1 = vertices[edge[0]]
        p2 = vertices[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]])
    ax.set_xlim(-size[0] / 2, size[0] / 2)
    ax.set_ylim(-size[1] / 2, size[1] / 2)
    plt.show()
    
def get_extrinsic_matrix(camera_position, camera_orientation):
    original_position = np.array([0, 0, 1])
    direction_vector = -camera_position
    a = math.acos(np.dot(original_position, camera_orientation)/np.linalg.norm(original_position)/np.linalg.norm(camera_orientation))
    cross_v = np.cross(camera_orientation, original_position)
    print(a/np.pi*180)
    print(cross_v)
    if np.linalg.norm(cross_v) == 0:
        x = y = z = 0
    else:
        x, y, z = cross_v / np.linalg.norm(cross_v)
    
    rotation_matrix = np.array([[math.cos(a) + x ** 2 * (1 - math.cos(a)), x * y * (1 - math.cos(a)) - z * math.sin(a), x * z * (1 - math.cos(a)) + y * math.sin(a)],
                                [y * x * (1 - math.cos(a)) + z * math.sin(a), math.cos(a) + y ** 2 * (1 - math.cos(a)), y * z * (1 - math.cos(a)) - x * math.sin(a)],
                                [z * x * (1 - math.cos(a)) - y * math.sin(a), z * y * (1 - math.cos(a)) + x * math.sin(a), math.cos(a) + z ** 2 * (1 - math.cos(a))]])
    _homo_matrix = np.eye(len(rotation_matrix) + 1)
    _homo_matrix[:len(rotation_matrix), :len(rotation_matrix)] = rotation_matrix
    rotation_matrix = _homo_matrix
    print(rotation_matrix)
    
    translation_matrix = np.eye(4)
    translation_matrix[:3, -1] = direction_vector
    print(translation_matrix)
    
    extrinsic_matrix = np.matmul(rotation_matrix, translation_matrix)
    extrinsic_matrix = np.where(np.abs(extrinsic_matrix) < 1e-13, 0, extrinsic_matrix)
    
    return extrinsic_matrix

def to_homogenous(points):
    homo_points = list()
    for point in points:
        point.append(1)
        homo_points.append(point)
    
    return homo_points

def from_homogenous(points):
    homo_points = list()
    for point in points:
        if np.abs(point[-1]) != 1.:
            raise ValueError("make homogenous vector first")
        homo_points.append(point[:-1])
    
    return homo_points

def get_intrinsic_matrix(focal_len, pixel_size):
    # princial point [0, 0]
    intrinsic_matrix = np.eye(3)
    for _i in range(len(pixel_size)):
        intrinsic_matrix[_i, _i] = focal_len * pixel_size[_i]
    
    return intrinsic_matrix

extrinsic_matrix = get_extrinsic_matrix(camera_position, camera_orientation)
intrinsic_matrix = get_intrinsic_matrix(focal_len, pixel_size)

vertices = to_homogenous(vertices)

_vertices = list()
for _vertice in vertices:
    _vertice = np.matmul(extrinsic_matrix, _vertice)[:3]
    _vertice = np.where(np.abs(_vertice) < 1e-13, 0, _vertice)
    _vertices.append(_vertice)
vertices_3d = _vertices

print(extrinsic_matrix)
print(vertices_3d)
show_3d(vertices_3d, edges)

print(intrinsic_matrix)
_vertices = list()
for _vertice in vertices_3d:
    _vertice = np.matmul(intrinsic_matrix, _vertice)
    _vertice = np.where(np.abs(_vertice) < 1e-13, 0, _vertice)
    if _vertice[-1] == 0:
         _vertice[-1] = 1e-13
    _vertice /= np.abs(_vertice[-1])
    _vertices.append(_vertice)
                
vertices_2d = _vertices
print(vertices_2d)
_vertices_2d = from_homogenous(vertices_2d)

show_2d(_vertices_2d, edges, pixel_size)