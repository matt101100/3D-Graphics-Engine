import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import math
from typing import List, Tuple

# define cube vertices, faces and normals
vertices = [
    [-1, -1, -1], # vertex 0
    [1, -1, -1],  # vertex 1
    [1, 1, -1],   # vertex 2
    [-1, 1, -1],  # vertex 3
    [-1, -1, 1],  # vertex 4
    [1, -1, 1],   # vertex 5
    [1, 1, 1],    # vertex 6
    [-1, 1, 1]    # vertex 7
]

faces = [
    (0, 1, 2, 3), # front
    (4, 5, 6, 7), # back
    (0, 3, 7, 4), # left
    (1, 2, 6, 5), # right
    (0, 1, 5, 4), # bottom
    (3, 2, 6, 7)  # top
]

faceNormals = [
    [0, 0, -1],  # front
    [0, 0, 1],   # back
    [-1, 0, 0],  # left
    [1, 0, 0],   # right
    [0, -1, 0],  # bottom
    [0, 1, 0]    # top
]

def loadObjectFile(filename: str):
    """
    Load an OBJ file and return vertices, normals, and faces.

    :param filename: path to the .obj file
    :return: tuple containing lists of vertices, normals, and faces
    """
    vertices = []
    normals = []
    faces = []
    # TODO: implement file loading and parsing

    return vertices, normals, faces

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
        [0, math.cos(angle / 2), math.sin(angle / 2), 0],
        [0, -math.sin(angle / 2), math.cos(angle / 2), 0],
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
        [math.cos(angle / 3), math.sin(angle / 3), 0, 0],
        [-math.sin(angle / 3), math.cos(angle / 3), 0 ,0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def computeLighting(faceNormal: List[float], faceCenter: np.ndarray) -> float:
    """
    Calculates intensity of lighting based on face normal and light direction.

    :param faceNormal: 1x3 normal given as a list of floats
    :param faceCenter: center coordinates of face with corresponding faceNormal
    :return: dot product between the light direction vector and faceNormal
             defaults to zero in cases where the dot product is negative
             as incoming light cannot be negative
    """
    # light comes from the camera at (0, 0, 0)
    lightDirection = -np.array(faceCenter)
    lightDirection = lightDirection / np.linalg.norm(lightDirection)
    dotProduct = np.dot(faceNormal, lightDirection)
    return max(dotProduct, 0) # clamp to 0 for normals facing away

def isFaceVisible(faceNormal: List[int], faceCenter: List[int],
                  cameraPos=[0, 0, 0]) -> bool:
    """
    Returns a bool based on whether a face with faceNormal can be seen from
    the current camera position to the faceCenter. Uses the dot product between
    the camera and face normal to compute how much the normal projects onto the
    view direction vector. Values greater than 0 mean that the face is visible.

    :param faceNormal: 1x3 normal vector given as a list of floats
    :param faceCenter: center coordinates of face with corresponding faceNormal
    :param cameraPos: position of the viewer / camera, defaults to (0, 0, 0)
    :return: a bool representing whether the face with given normal and center
             coordinates is visible from the camera at cameraPos
    """
    viewDirection = cameraPos - np.array(faceCenter)
    dotProduct = np.dot(faceNormal, viewDirection)
    return dotProduct > 0 # only True when face is visible

def drawCube(rotationAngle: float) -> None:
    """
    Draws a solid cube to the screen with lighting.
    Computes position of each vertex after rotation and translation into 
    the z-axis and uses back-face culling to hide faces that are 
    not visible from the current camera position.

    :param: rotationAngle: the angle through which vertices are rotated
    """
    glBegin(GL_QUADS)
    # set up rotation and perspective matrices
    # note that we combine the rotation matrices such that on each frame
    # we rotate first in the x direction, then y, then z
    rotationMatrixY = generateRotationMatrixY(rotationAngle)
    rotationMatrixX = generateRotationMatrixX(rotationAngle)
    rotationMatrixZ = generateRotationMatrixZ(rotationAngle)
    combinedRotationMatrix = rotationMatrixX.dot(rotationMatrixY).dot(rotationMatrixZ)
    perspectiveMatrix = generatePerspectiveMatrix(np.radians(45), 4/3, 0.1, 50.0)

    for face, normal in zip(faces, faceNormals):
        # rotate the normal
        rNormal = combinedRotationMatrix.dot(normal + [0])[:3]

        # rotate cube face vertices
        faceVertices = [vertices[i] for i in face]
        rotatedVertices = [combinedRotationMatrix.dot(np.append(v, 1))[:3] for v in faceVertices]

        # compute the face center by averaging rotated vertices
        faceCenter = np.mean(rotatedVertices, axis=0)
        faceCenter[2] -= 5 # move cube back along the z-axis

        # check if face is visible
        if (isFaceVisible(rNormal, faceCenter)):
            # handle lighting for visible faces
            lightingIntensity = computeLighting(rNormal, faceCenter)

            color = lightingIntensity # scale color brightness with intensity
            glColor3f(color, color, color)

            # project vertices only if the face is visible
            for i in range(len(face)):
                # get edge vertices
                rVertex1 = np.append(rotatedVertices[i], 1)
                rVertex2 = np.append(rotatedVertices[(i + 1) % 4], 1)

                # translate rotated vertices the same amount as faceCenter
                rVertex1[2] -= 5
                rVertex2[2] -= 5

                """
                !!! Note !!! 
                The values faceCenter[2], rVertex1[2] and rVertex2[2]
                need to be equal or else there is a slight mismatch between
                vertices and normal, leading to janky visible --> invisible
                edge transitions as objects rotate / move
                """

                # apply perspective projection matrix
                projectedVertex1 = perspectiveMatrix.dot(rVertex1)
                projectedVertex2 = perspectiveMatrix.dot(rVertex2)

                # apply perspective division
                projectedVertex1 /= projectedVertex1[3]
                projectedVertex2 /= projectedVertex2[3]

                # draw edges of visible faces
                glVertex3f(*projectedVertex1[:3])
                glVertex3f(*projectedVertex2[:3])
    glEnd()

def drawObject(vertices: List[List[float]], faces: List[List[int]], 
               normals: List[List[float]], 
               rotationAngles: Tuple[float, float, float]) -> None:
    """
    Draw a loaded OBJ model with lighting and perspective projection.

    :param vertices: list of vertex positions
    :param faces: list of faces (vertex indices)
    :param normals: list of normals
    :param rotationAngles: tuple of rotation angles (x, y, z)
    """
    pass

def main():
    """
    Initialises Pygame modules, setting up frameworks for graphics and event
    handling. Sets up display window for rendering with the given display,
    OpenGL and double buffering.

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

    # begin rendering loop
    running = True
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
        
        # allow for rotation but prevent the angle value from getting too big
        rotationAngle = (rotationAngle + 0.02) % (6 * math.pi)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        drawCube(rotationAngle)
        pygame.display.flip()

        clock.tick(60) # cap framerate

    pygame.quit()

if __name__ == "__main__":
    main()