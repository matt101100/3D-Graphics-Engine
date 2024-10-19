import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import math
from typing import List, Tuple

# define cube vertices and edges
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
    viewDirection = cameraPos - np.array(faceCenter)
    dotProduct = np.dot(faceNormal, viewDirection)
    return dotProduct > 0 # only True when face is visible

"""
Draws a cube wireframe to the screen
Computes position of each vertex after rotation and translation into the z-axis
and uses back-face culling to hide faces that are not visible from the
current camera position
"""
def drawCube(rotationAngle: float) -> None:
    glBegin(GL_LINES)
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
        faceCenter[2] -= 10 # move cube back along the z-axis

        # check if face is visible
        if (isFaceVisible(rNormal, faceCenter)):
            # len(face) project vertices only if the face is visible
            for i in range(4):
                # get edge vertices
                rVertex1 = np.append(rotatedVertices[i], 1)
                rVertex2 = np.append(rotatedVertices[(i + 1) % 4], 1)
                rVertex1[2] -= 10
                rVertex2[2] -= 10

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