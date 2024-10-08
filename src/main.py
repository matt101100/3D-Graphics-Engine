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
def loadPerspectiveMatrix(fov, aspect, near, far):
    f = 1 / math.tan(fov / 2)
    depth = far - near
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / depth, -2 * far * near / depth],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def main():
    # todo:
    return

if __name__ == "__main__":
    main()