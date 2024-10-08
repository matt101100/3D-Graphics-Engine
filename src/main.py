import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
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

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# creates a perspective projection matrix
def generatePerspectiveMatrix(fov, aspect, near, far):
    f = 1 / math.tan(fov / 2)
    depth = far - near
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / depth, -2 * far * near / depth],
        [0, 0, -1, 0]
    ], dtype=np.float32)

# creates a rotation matrix handling rotations around the y axis
def generateRotationMatrixY(angle):
    return np.array([
        [math.cos(angle), 0, math.sin(angle), 0],
        [0, 1, 0, 0],
        [-math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# performs matrix multiplication on a vertex to update its position
# returns results as 3D coordinates
def transformVertex(vertex, matrix):
    v = np.array(vertex + [1]) # convert to homogenous coordinates
    transformed = matrix.dot(v)
    return transformed[:3] / transformed[3] # convert back to 3D coords

def drawCube(rotationAngle):
    glBegin(GL_LINES)
    rotationMatrix = generateRotationMatrixY(rotationAngle)
    perspectiveMatrix = generatePerspectiveMatrix(np.radians(45), 4/3, 0.1, 50.0)

    for edge in edges:
        for vertexIdx in edge:
            vertex = vertices[vertexIdx] + [1]
            vertex = np.array(vertex, dtype=np.float32)
            rotatedVertex = rotationMatrix.dot(vertex)
            projectedVertex = perspectiveMatrix.dot(rotatedVertex)
            if (projectedVertex[3] != 0):
                projectedVertex = projectedVertex[:3] / projectedVertex[3]
            glVertex3f(*projectedVertex)
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    clock = pygame.time.Clock()
    rotationAngle = 0

    running = True
    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False
        
        rotationAngle += 0.02
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        drawCube(rotationAngle)
        pygame.display.flip()

        clock.tick(60) # cap framerate

    pygame.quit()

if __name__ == "__main__":
    main()