import pygame
from pygame.locals import *
from OpenGL.GL import *
from PIL import Image
import numpy as np
import math
from typing import List, Tuple

class Camera:
    """
    Represents a camera object to allow the user to observe 3D space. Packages
    position, target and up vectors and handles camera functionality using
    look-at matrix principles. Also handles W, A, S, D and mouse inputs to
    simulate motion in a 3D space.
    """

    def __init__(self, position: List[float], target: List[float],
                 up: List[float], speed: float, sens: float):
        """
        Initialises the Camera object.

        :param position: the camera's initial position in 3D space
        :param target: the point the camera is aimed at
        :param up: the 'up' direction vector, defines the camera's orientation
        """
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.speed = speed
        self.sens = sens
    
    @property
    def forward(self):
        """
        Computes the forward vector using current camera target and position.

        :return: the updated forward vector
        """
        forward = self.target - self.position
        return forward / np.linalg.norm(forward)
    
    @property
    def right(self):
        """
        Computes the right vector using current up and forward vectors.

        :return: the updated forward vector
        """
        right = np.cross(self.forward, self.up)
        return right / np.linalg.norm(right)

    @staticmethod
    def generate_look_at_matrix(position, target, up):
        """
        Constructs the look-at matrix using position, target and up vectors.

        :param position: current camera position in 3D sppace
        :param target: the point the camera is aimed at
        :param up: up direction vector, defines the camera's orientation
        :return: the 4x4 look-at matrix that converts points into the new 
                 observation vector space
        """
        
        # generate vectors that define the new space
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)

        # construct the look-at matrix
        look_at = np.array([
            [right[0], new_up[0], -forward[0], 0],
            [right[1], new_up[1], -forward[1], 0],
            [right[2], new_up[2], -forward[2], 0],
            [-np.dot(right, position), -np.dot(new_up, position),
             np.dot(forward, position), 1]
        ], dtype=np.float32)

        return look_at
    
    def move(self, direction: str) -> None:
        if (direction == "FORWARD"):
            # move the camera forward
            self.position += self.speed * self.forward
            self.target += self.speed * self.forward
        
        elif (direction == "BACKWARD"):
            # move the camera backward
            self.position -= self.speed * self.forward
            self.target -= self.speed * self.forward

        elif (direction == "LEFT"):
            # move the camera to the left
            self.position -= self.speed * self.right
            self.target -= self.speed * self.right
        
        elif (direction == "RIGHT"):
            # move the camera to the right
            self.position += self.speed * self.right
            self.target += self.speed * self.right
    
    def get_look_at_matrix(self):
        """
        Gets the look-at matrix.

        :return: the generated look-at matrix.
        """
        return self.generate_look_at_matrix(self.position, self.target, self.up)

def handle_movement(keys, camera: Camera) -> None:
    if (keys[pygame.K_w]):
        camera.move("FORWARD")
    elif (keys[pygame.K_s]):
        camera.move("BACKWARD")
    elif (keys[pygame.K_a]):
        camera.move("LEFT")
    elif (keys[pygame.K_d]):
        camera.move("RIGHT")

def load_object_file(file_name: str) -> Tuple[List[List[float]],
                                              List[List[float]],
                                              List[Tuple[List[int], List[int], 
                                                   List[int]]],
                                              List[List[float]]]:
    """
    Load an OBJ file and return vertices, normals, faces and texture coordinates.

    :param file_name: path to the .obj file
    :return: tuple containing lists of vertices, normals, faces, and 
             texture coordinates, faces are returned as a tuple of vertex indices, normal indices, and texture indices.
    """
    vertices = []
    normals = []
    faces = []
    texture_coords = []

    try:
        with open(file_name, 'r') as f:
            for line in f:
                words = line.split()
                if (len(words) == 0):
                    continue

                if (words[0] == "v"):
                    # parse vertex lines
                    temp = []
                    for s in words[1:4]:
                        temp.append(float(s))
                    vertices.append(temp)

                elif (words[0] == "vn"):
                    # parse normal lines
                    temp = []
                    for s in words[1:4]:
                        temp.append(float(s))
                    normals.append(temp)
                
                elif (words[0] == "vt"):
                    # parse texture coordinate lines
                    temp = []
                    for s in words[1:3]:
                        temp.append(float(s))
                    texture_coords.append(temp)

                elif (words[0] == "f"):
                    # parse face lines -- get indices corresponding to vertices
                    # normals and textures for this face
                    face_vertex_indices = []
                    face_normal_indices = []
                    face_texture_indices = []
                    for s in words[1:]:
                        # each line has form:
                        # vertex/texture/normal or vertex//normal
                        indices = s.split('/')
                        vertex_idx = int(indices[0]) - 1 # convert to 0-indexing
                        face_vertex_indices.append(vertex_idx)

                        # texture indices
                        if (len(indices) > 1 and indices[1]):
                            tex_idx = int(indices[1]) - 1
                            face_texture_indices.append(tex_idx)

                        # normal indices
                        if (len(indices) > 2 and indices[2]):
                            # line specifies normals as well
                            normal_idx = int(indices[2]) - 1
                            face_normal_indices.append(normal_idx)
                    
                    faces.append((face_vertex_indices, face_normal_indices,
                                   face_texture_indices))
                 
    except FileNotFoundError:
        print("Error: Invalid file path provided.")
        return None, None, None, None
    
    return vertices, normals, faces, texture_coords


def load_texture(image_path: str) -> int:
    # TODO: implement texture file loading from png or jpeg formats
    pass

def generate_perspective_matrix(fov: float, aspect: float, near: float, 
                              far: float) -> np.ndarray:
    """
    Generates a perspective matrix based on the given fov, aspect ratio and
    near and far plane locations in the z-axis.

    :param fov: camera field of view
    :param aspect: desired image aspect ratio
    :param near: z-axis value representing the near plane
    :param far: z-axis value representing the far plane
    :return: numpy 4x4 perspective projection matrix with 32-bit float elements
    """
    f = 1 / math.tan(fov / 2)
    depth = far - near
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / depth, -2 * far * near / depth],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def generate_rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Generates a 4x4 rotation matrix in the x direction.

    :param angle: angle through which to rotate a given vertex
    :return: numpy 4x4 rotation matrix
    """
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle), math.sin(angle), 0],
        [0, -math.sin(angle), math.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def generate_rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Generates a 4x4 rotation matrix in the y direction.

    :param angle: angle through which to rotate a given vertex
    :return: numpy 4x4 rotation matrix
    """
    return np.array([
        [math.cos(angle), 0, math.sin(angle), 0],
        [0, 1, 0, 0],
        [-math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def generate_rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Generates a 4x4 rotation matrix in the z direction.

    :param angle: angle through which to rotate a given vertex
    :return: numpy 4x4 rotation matrix
    """
    return np.array([
        [math.cos(angle), math.sin(angle), 0, 0],
        [-math.sin(angle), math.cos(angle), 0 ,0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def compute_lighting(face_normal: List[float]) -> float:
    """
    Calculates intensity of lighting based on face normal and light direction.

    :param face_normal: 1x3 normal given as a list of floats
    :return: dot product between the light direction vector and face_normal
             defaults to zero in cases where the dot product is negative
             as incoming light cannot be negative
    """
    # assume light comes from the camera along the z-axis
    light_direction = np.array([0, 0, 1])
    light_direction = light_direction / np.linalg.norm(light_direction)
    dot_prod = np.dot(face_normal, light_direction)
    return max(dot_prod, 0) # clamp to 0 for normals facing away

def is_face_visible(face_normal: List[float], camera_direction=[0, 0, 1]):
    """
    Returns a bool based on whether a face with face_normal can be seen from
    the current camera position. Uses the dot product between the camera and 
    face normal to compute how much the normal projects onto the
    view direction vector.

    :param face_normal: 1x3 normal vector given as a list of floats
    :param camera_direction: direction the camera faces, defaults to (0, 0, 0)
    :return: a bool representing whether the face with given normal is visible 
             from the camera at cameraPos
    """
    dot_prod = np.dot(face_normal, camera_direction)
    return dot_prod > 0 # only true when face is visible

def compute_normal(v0: List[float], v1: List[float],
                   v2: List[float]) -> List[float]:
    """
    Compute the normal for a triangle given its three vertices.

    :param v0: first vertex of the triangle
    :param v1: second vertex of the triangle
    :param v2: third vertex of the triangle
    :return: normal vector of the triangle
    """
    edge1 = np.array(v1) - np.array(v0)
    edge2 = np.array(v2) - np.array(v0)
    normal = np.cross(edge1, edge2) # cross product to compute orthogonal vecs
    normal /= np.linalg.norm(normal)  # normalize the normal vector
    return normal

def draw_object(vertices: List[List[float]], faces: List[List[int]], 
               normals: List[List[float]], 
               rotation_angles: Tuple[float, float, float],
               camera: Camera) -> None:
    """
    Draw a loaded OBJ model with lighting and perspective projection, computing
    normals and visibility for each triangle separately.

    :param vertices: list of vertex positions
    :param faces: list of faces (vertex indices)
    :param normals: list of normals (not currently used by the function)
    :param rotation_angles: tuple of rotation angles (x, y, z)
    """
    # set up rotation matrices
    angle_x, angle_y, angle_z = rotation_angles
    rotation_matrix_x = generate_rotation_matrix_x(angle_x)
    rotation_matrix_y = generate_rotation_matrix_y(angle_y / 2)
    rotation_matrix_z = generate_rotation_matrix_z(angle_z / 3)
    combined_rotation = rotation_matrix_x.dot(
                        rotation_matrix_y).dot(rotation_matrix_z)

    # generate perspective projection matrix and look-at matrix
    perspective_matrix = generate_perspective_matrix(np.radians(45), 
                                                    4/3, 0.1, 50.0)
    look_at_matrix = camera.get_look_at_matrix()

    # combine the perspective and look-at matrices for a total view matrix
    view_matrix = perspective_matrix.dot(look_at_matrix) # view x perspective

    # begin render
    glBegin(GL_TRIANGLES)
    for face, normal_indices, texture_indices in faces:
        # get the vertices for the current triangle
        v0, v1, v2 = [vertices[i] for i in face]
        rv0 = combined_rotation.dot(np.append(v0, 1))[:3]
        rv1 = combined_rotation.dot(np.append(v1, 1))[:3]
        rv2 = combined_rotation.dot(np.append(v2, 1))[:3]

        # apply camera view transform
        cv0 = view_matrix.dot(np.append(rv0, 1))
        cv1 = view_matrix.dot(np.append(rv1, 1))
        cv2 = view_matrix.dot(np.append(rv2, 1))

        if (normal_indices):
            # use normals provided in the .obj file
            n0, n1, n2 = [normals[i] for i in normal_indices]
            rn0 = combined_rotation.dot(np.append(n0, 1))[:3]
            rn1 = combined_rotation.dot(np.append(n1, 1))[:3]
            rn2 = combined_rotation.dot(np.append(n2, 1))[:3]
            normal = np.mean([rn0, rn1, rn2], axis=0)
        else:
            # compute the normal for this triangle if not supplied in the .obj
            normal = compute_normal(cv0[:3], cv1[:3], cv2[:3])

        # check if the triangle is visible
        if is_face_visible(normal):
            # compute lighting for this triangle
            lighting_intensity = compute_lighting(normal)
            color = lighting_intensity
            glColor3f(color, color, color)

            # project and draw each vertex
            for v in [cv0, cv1, cv2]:
                # apply perspective projection
                # r_vertex = np.append(v, 1)
                # r_vertex[2] -= 30
                projected_vertex = v / v[3]
                # projected_vertex /= projected_vertex[3]
                
                glVertex3f(*projected_vertex[:3])
    glEnd()

def main():
    """
    Initialises Pygame modules, setting up frameworks for graphics and event
    handling. Sets up display window for rendering with the given display,
    OpenGL and double buffering. Loads object file.

    Rendering loop calls functions to display objects and Pygame handles user
    events, specifically when quit conditions are met, such as by pressing
    the 'x' on the display window.

    Framerate is capped at 60 fps.
    """
    # init Pygame and OpenGL settings
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    clock = pygame.time.Clock()
    rotation_angle = 0

    # load object file
    # TODO: accept file paths from the user
    vertices, normals, faces, tex_coords = load_object_file("objects/axis.obj")
    if (vertices == None and normals == None and faces == None 
        and tex_coords == None):
        # invalid filepath end point
        print("Exiting...")
        return 1
    elif (len(vertices) == 0 or len(faces) == 0):
        # invalid file contents end point
        print("Invalid .obj file defines no vertices or faces.")
        print("Exiting...")
        return 1

    # initialise camera
    move_speed = 0.03
    mouse_sens = 0.001
    camera = Camera([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], move_speed, mouse_sens)

    # enable depth testing
    # --> possibly implement this myself: depth buffering or painter's algo
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # begin rendering loop
    running = True
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
            
            keys = pygame.key.get_pressed()
            handle_movement(keys, camera)
        
        # allow for rotation but prevent the angle value from getting too big
        # rotation_angle = (rotation_angle + 0.02) % (6 * math.pi)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # drawCube(rotation_angle)
        draw_object(vertices, faces, normals,
                    (rotation_angle, rotation_angle, rotation_angle), camera)
        pygame.display.flip()

        clock.tick(60) # cap framerate

    pygame.quit()

if __name__ == "__main__":
    main()