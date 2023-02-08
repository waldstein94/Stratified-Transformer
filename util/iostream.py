# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/indoor.py
CUBOID_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (255., 152., 150.),
    2: (174., 199., 232.),
    3: (152., 223., 138.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (31., 119., 180.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (200., 54., 131.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (0., 0., 0.),
    28: (158., 218., 229.),
}


def return_depth(dpath):
    return np.asarray(o3d.io.read_image(dpath))


def viz_depth(dpath):
    depth_raw = o3d.io.read_image(dpath)
    plt.imshow(depth_raw)
    plt.imsave('test.png', depth_raw)


def load_obj_mesh(mesh_file):
    """
    :param mesh_file: path of point cloud
    return vertices, faces, colors
    """
    vertex_data = []
    color_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            if len(values) > 5:
                c = list(map(float, values[4:]))
                color_data.append(c)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if len(color_data) > 0:
        colors = np.array(color_data)
    else:
        colors = None

    return vertices, faces, colors


def save_obj(out, sample, color=None):
    with open(out, 'w') as file:
        if color is None:
            for v1 in sample:
                file.write('v %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2]))
        else:
            for (v1, c) in zip(sample, color):
                file.write(
                    'v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2], c[0], c[1], c[2]))  # 0.5*c,0.5*c,0.5*


def save_obj_color_coding(out, samples, labels):
    with open(out, 'w') as file:
        for (v, l) in zip(samples, labels):
            c = CUBOID_COLOR_MAP[l]
            file.write(
                'v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))


def save_obj_line(out, end_pts1, end_pts2):
    with open(out, 'w') as file:
        for i, (v1, v2) in enumerate(zip(end_pts1, end_pts2)):
            # if np.linalg.norm(v1-v2)>2.0:
            #     continue
            file.write('v %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2]))
            file.write('v %.4f %.4f %.4f\n' % (v2[0], v2[1], v2[2]))

        for i in range(len(end_pts1)):
            file.write('l %d %d\n' % ((i * 2 + 1, i * 2 + 2)))


def save_offset(out, pts, cls, angle, scale):
    # zero position for each face
    lookup = [0, 1, 2, 2, 1, 0]
    zeros = np.zeros(len(cls)).astype(int)
    # pred to lookup
    for i, idx in enumerate(lookup):
        zeros[cls==i] = idx
    # angle and scale to offset, -170~170
    rad = np.deg2rad(angle*20-180+10)
    offsets = np.asarray([scale*np.cos(rad), scale*np.sin(rad)]).T
    with open(out, 'w') as file:
        for i, (p, o, idx) in enumerate(zip(pts, offsets, zeros)):
            o = np.insert(o, idx, 0)
            v1 = p+o
            file.write('v %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2]))


    # fill zero with index