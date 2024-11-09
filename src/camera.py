import pygame
from pygame.locals import *
import numpy as np
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
    
    @property
    def new_up(self):
        """
        Computes the new up vector based on forward and right.

        :return: the updated up vector
        """
        new_up = np.cross(self.right, self.forward)
        return new_up / np.linalg.norm(new_up)

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

    # def move(self, direction: str) -> None:
    #     if (direction == "FORWARD"):
    #         # move the camera forward
    #         self.position[2] += self.speed * self.target[2]

    #     elif (direction == "BACKWARD"):
    #         # move the camera backward
    #         self.position[2] -= self.speed * self.target[2]

    #     elif (direction == "LEFT"):
    #         # move the camera to the left
    #         self.position -= self.speed * self.right
        
    #     elif (direction == "RIGHT"):
    #         # move the camera to the right
    #         self.position += self.speed * self.right
        
    #     # self.target = self.position + self.forward ## !! FIX THIS !! make it sot that look direction and target are different things, look dir hardcode as class variable but target needs update on each movement
    
    def get_look_at_matrix(self):
        """
        Gets the look-at matrix.

        :return: the generated look-at matrix.
        """
        return self.generate_look_at_matrix(self.position, self.target, self.up)
    
    def move(self, keys, delta_time):
        """
        Updates the camera position based on key inputs and adjusts target 
        for forward and strafe movement.

        :param keys: the key states (from Pygame's key.get_pressed())
        :param delta_time: the time elapsed since the last frame
        """
        # movement speed scaled by delta_time
        velocity = self.speed * delta_time

        # forward and backward movement
        if keys[pygame.K_w]:  # forward
            self.position += self.forward * velocity
            # self.target += self.forward * velocity
        if keys[pygame.K_s]:  # backward
            self.position -= self.forward * velocity
            # self.target -= self.forward * velocity

        # right and left strafe movement
        if keys[pygame.K_d]:  # right
            self.position += self.right * velocity
            # self.target += self.right * velocity
        if keys[pygame.K_a]:  # left
            self.position -= self.right * velocity
            # self.target -= self.right * velocity
        
        self.target = self.position + self.forward