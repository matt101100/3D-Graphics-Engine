import pygame
from pygame.locals import *
from OpenGL.GL import *
from PIL import Image
import numpy as np
import math
from typing import List, Tuple

def loadObjectFile(filename: str) -> Tuple[List[List[float]], List[List[float]], 
                                           List[Tuple[List[int], List[int], List[int]]], 
                                           List[List[float]]]:
    """
    Load an OBJ file and return vertices, normals, faces and texture coordinates.

    :param filename: path to the .obj file
    :return: tuple containing lists of vertices, normals, faces, and 
             texture coordinates.
             Faces are returned as a tuple of vertex indices, normal indices, 
             and texture indices.
    """
    vertices = []
    normals = []
    faces = []
    textureCoords = []

    try:
        with open(filename, 'r') as f:
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
                    textureCoords.append(temp)

                elif (words[0] == "f"):
                    # parse face lines -- get indices corresponding to vertices
                    # normals and textures for this face
                    faceVertexIndices = []
                    faceNormalIndices = []
                    faceTextureIndices = []
                    for s in words[1:]:
                        # each line has form vertex/texture/normal or vertex//normal
                        indices = s.split('/')
                        vertexIdx = int(indices[0]) - 1 # convert to 0-indexing
                        faceVertexIndices.append(vertexIdx)

                        # texture indices
                        if (len(indices) > 1 and indices[1]):
                            texIdx = int(indices[1]) - 1
                            faceTextureIndices.append(texIdx)

                        # normal indices
                        if (len(indices) > 2 and indices[2]):
                            # line specifies normals as well
                            normalIdx = int(indices[2]) - 1
                            faceNormalIndices.append(normalIdx)
                    
                    faces.append((faceVertexIndices, faceNormalIndices,
                                   faceTextureIndices))
                 
    except FileNotFoundError:
        print("Error: Invalid file path provided.")
        return None, None, None, None
    
    return vertices, normals, faces, textureCoords


def loadTexture(imagePath: str) -> int:
    # TODO: implement texture file loading from png or jpeg formats
    
    pass

def generatePerspectiveMatrix(fov: float, aspect: float, near: float, 
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

def generateRotationMatrixX(angle: float) -> np.ndarray:
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

def generateRotationMatrixY(angle: float) -> np.ndarray:
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

def generateRotationMatrixZ(angle: float) -> np.ndarray:
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

def computeLighting(faceNormal: List[float]) -> float:
    """
    Calculates intensity of lighting based on face normal and light direction.

    :param faceNormal: 1x3 normal given as a list of floats
    :return: dot product between the light direction vector and faceNormal
             defaults to zero in cases where the dot product is negative
             as incoming light cannot be negative
    """
    # assume light comes from the camera along the z-axis
    lightDirection = np.array([0, 0, 1])
    lightDirection = lightDirection / np.linalg.norm(lightDirection)
    dotProduct = np.dot(faceNormal, lightDirection)
    return max(dotProduct, 0) # clamp to 0 for normals facing away

def isFaceVisible(faceNormal: List[float], cameraDirection=[0, 0, 1]):
    """
    Returns a bool based on whether a face with faceNormal can be seen from
    the current camera position. Uses the dot product between the camera and 
    face normal to compute how much the normal projects onto the
    view direction vector. Values greater than 0 mean that the face is visible.

    :param faceNormal: 1x3 normal vector given as a list of floats
    :param cameraDirection: direction the camera faces, defaults to (0, 0, 0)
    :return: a bool representing whether the face with given normal is visible 
             from the camera at cameraPos
    """
    dotProduct = np.dot(faceNormal, cameraDirection)
    return dotProduct > 0 # only true when face is visible

def computeNormal(v0: List[float], v1: List[float], v2: List[float]) -> List[float]:
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

def drawObject(vertices: List[List[float]], faces: List[List[int]], 
               normals: List[List[float]], 
               rotationAngles: Tuple[float, float, float]) -> None:
    """
    Draw a loaded OBJ model with lighting and perspective projection, computing
    normals and visibility for each triangle separately.

    :param vertices: list of vertex positions
    :param faces: list of faces (vertex indices)
    :param normals: list of normals (not currently used by the function)
    :param rotationAngles: tuple of rotation angles (x, y, z)
    """
    # set up rotation matrices
    angleX, angleY, angleZ = rotationAngles
    rotationMatrixX = generateRotationMatrixX(angleX)
    rotationMatrixY = generateRotationMatrixY(angleY / 2)
    rotationMatrixZ = generateRotationMatrixZ(angleZ / 3)
    combinedRoMatrix = rotationMatrixX.dot(rotationMatrixY).dot(rotationMatrixZ)

    # generate perspective projection matrix
    perspectiveMatrix = generatePerspectiveMatrix(np.radians(45), 4/3, 0.1, 50.0)

    # begin render
    glBegin(GL_TRIANGLES)
    for face, normalIndices, textureIndices in faces:
        # get the vertices for the current triangle
        v0, v1, v2 = [vertices[i] for i in face]
        rv0 = combinedRoMatrix.dot(np.append(v0, 1))[:3]
        rv1 = combinedRoMatrix.dot(np.append(v1, 1))[:3]
        rv2 = combinedRoMatrix.dot(np.append(v2, 1))[:3]

        if (normalIndices):
            # use normals provided in the .obj file
            n0, n1, n2 = [normals[i] for i in normalIndices]
            rn0 = combinedRoMatrix.dot(np.append(n0, 1))[:3]
            rn1 = combinedRoMatrix.dot(np.append(n1, 1))[:3]
            rn2 = combinedRoMatrix.dot(np.append(n2, 1))[:3]
            normal = np.mean([rn0, rn1, rn2], axis=0)
        else:
            # compute the normal for this triangle if not supplied in the .obj
            normal = computeNormal(rv0, rv1, rv2)

        # check if the triangle is visible
        if isFaceVisible(normal):
            # compute lighting for this triangle
            lightingIntensity = computeLighting(normal)
            color = lightingIntensity
            glColor3f(color, color, color)

            # project and draw each vertex
            for v in [rv0, rv1, rv2]:
                # apply perspective projection
                rVertex = np.append(v, 1)
                rVertex[2] -= 10
                projectedVertex = perspectiveMatrix.dot(rVertex)
                projectedVertex /= projectedVertex[3]
                
                glVertex3f(*projectedVertex[:3])
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
    rotationAngle = 0

    # load object file
    # TODO: accept file paths from the user
    vertices, normals, faces, texCoords = loadObjectFile("objects/ship.obj")
    if (vertices == None and normals == None and faces == None 
        and texCoords == None):
        # invalid filepath end point
        print("Exiting...")
        return 1
    elif (len(vertices) == 0 or len(faces) == 0):
        # invalid file contents end point
        print("Invalid .obj file defines no vertices or faces.")
        print("Exiting...")
        return 1

    # enable depth testing --> possibly implement this myself: depth buffering or painter's algo
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # begin rendering loop
    running = True
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
        
        # allow for rotation but prevent the angle value from getting too big
        rotationAngle = (rotationAngle + 0.02) % (6 * math.pi)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # drawCube(rotationAngle)
        drawObject(vertices, faces, normals, (rotationAngle, rotationAngle, rotationAngle))
        pygame.display.flip()

        clock.tick(60) # cap framerate

    pygame.quit()

if __name__ == "__main__":
    main()