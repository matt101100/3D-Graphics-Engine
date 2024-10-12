import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import math

# define cube vertices and edges
vertices = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
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

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

"""
Generates a perspective matrix based on the given fov, aspect ratio and
near and far plane locations in the z-axis
"""
def generatePerspectiveMatrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1 / math.tan(fov / 2)
    depth = far - near
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / depth, -2 * far * near / depth],
        [0, 0, -1, 0]
    ], dtype=np.float32)

"""
The following 3 functions generate rotation matrices
for multiplying a position vector to compute position after rotation by angle
"""
def generateRotationMatrixX(angle: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle / 2), math.sin(angle / 2), 0],
        [0, -math.sin(angle / 2), math.cos(angle / 2), 0],
        [0, 0, 0, 1]
    ])

def generateRotationMatrixY(angle: float) -> np.ndarray:
    return np.array([
        [math.cos(angle), 0, math.sin(angle), 0],
        [0, 1, 0, 0],
        [-math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def generateRotationMatrixZ(angle: float) -> np.ndarray:
    return np.array([
        [math.cos(angle / 3), math.sin(angle / 3), 0, 0],
        [-math.sin(angle / 3), math.cos(angle / 3), 0 ,0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

"""
Returns a bool based on whether a face with faceNormal can be seen from
the current camera position to the faceCenter
"""
def isFaceVisible(faceNormal: list[int], faceCenter: list[int], cameraPos=[0, 0, 0]) -> bool:
    viewDirection = -np.array(faceCenter)
    dotProduct = np.dot(faceNormal, viewDirection)
    return dotProduct > 0 # only True when face is visible

"""
Draws a cube wireframe to the screen.
Computes position of each vertex after rotation and translation into the z-axis
and uses back-face culling to hide faces that are not visible from the
current camera position
"""
def drawCube(rotationAngle: float) -> None:
    glBegin(GL_LINES)
    rotationMatrixY = generateRotationMatrixY(rotationAngle)
    rotationMatrixX = generateRotationMatrixX(rotationAngle)
    rotationMatrixZ = generateRotationMatrixZ(rotationAngle)
    perspectiveMatrix = generatePerspectiveMatrix(np.radians(45), 4/3, 0.1, 50.0)

    for edge in edges:
        for vertexIdx in edge:
            # convert 3D coordinates into 4D homogenous coordinates
            vertex = vertices[vertexIdx] + [1]
            vertex = np.array(vertex, dtype=np.float32)

            rotatedVertex = rotationMatrixX.dot(vertex)
            rotatedVertex = rotationMatrixY.dot(rotatedVertex)
            rotatedVertex = rotationMatrixZ.dot(rotatedVertex)
            rotatedVertex[2] -= 10 # shift the cube along the z-axis away from the screen

            projectedVertex = perspectiveMatrix.dot(rotatedVertex)
            # perspective division
            if (projectedVertex[3] != 0):
                projectedVertex = projectedVertex[:3] / projectedVertex[3]
            
            glVertex3f(*projectedVertex) # draws lines between two vertices
    glEnd()

def main():
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