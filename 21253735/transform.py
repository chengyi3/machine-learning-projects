
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import math
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *
# def doesarmpassthroughobstacles(armpos, obstacles):
#
#     for arm in armpos:
#         for obstacle in obstacles:
#             start = arm[0]
#             end = arm[1]
#             s = np.asarray(start)
#             e = np.asarray(end)
#             startx = start[0]
#             starty = start[1]
#             endx = end[0]
#             endy = end[1]
#             k = [obstacle[0], obstacle[1]]
#             endv = [endx, endy]
#             startv = [startx,starty]
#             minx = min(startx, endx)
#             maxx = max(startx, endx)
#             miny = min(starty,endy)
#             maxy = max(starty, endy)
#             d = np.linalg.norm(np.cross(e-s, s - k))/np.linalg.norm(e-s)
#             if (d <= obstacle[2]):
#                 if (obstacle[0] >= minx and obstacle[0] <= maxx) and (obstacle[1] >= miny and obstacle[1] <= maxy):
#                     return True
#
#
#     return False
            # if (obstacle[0] > minx and obstacle[0] < maxx)

def transformToMaze(arm, goals, obstacles, window, granularity):
    # """This function transforms the given 2D map to the maze in MP1.
    #
    #     Args:
    #         arm (Arm): arm instance
    #         goals (list): [(x, y, r)] of goals
    #         obstacles (list): [(x, y, r)] of obstacles
    #         window (tuple): (width, height) of the window
    #         granularity (int): unit of increasing/decreasing degree for angles
    #
    #     Return:
    #         Maze: the maze instance generated based on input arguments.

    # """
    armlimit = arm.getArmLimit()
    range1 = armlimit[0]
    range2 = armlimit[1]
    offsets = [range1[0], range2[0]]
    numberofrows = int((range1[1]-range1[0])/granularity) + 1
    numberofcolumns = int((range2[1]-range2[0])/granularity) + 1
    input_map = []
    for i in range(0, numberofrows):
        input_map.append([SPACE_CHAR]*numberofcolumns)

    initial_alpha = arm.getArmAngle()[0]
    initial_belta = arm.getArmAngle()[1]
    angles = (initial_alpha,initial_belta)
    initial_pos = angleToIdx(angles, offsets, granularity )
    input_map[initial_pos[0]][initial_pos[1]] = START_CHAR
    # initial_row = math.floor((initial_alpha - range1[0])/granularity) + 1
    # initial_column = math.floor((initial_belta - range2[0])/granularity) + 1
    # input_map[initial_pos[0]][initial_pos[1]] = START_CHAR
    # print(START_CHAR)

    for row in range(0, numberofrows):
        for column in range(0, numberofcolumns):
            index = (row, column)
            cur_angle = idxToAngle(index, offsets, granularity)
            arm.setArmAngle(cur_angle)
            if doesArmTouchObstacles(arm.getArmPos(), obstacles):
                input_map[row][column] = WALL_CHAR
            if not isArmWithinWindow(arm.getArmPos(), window):
                input_map[row][column] = WALL_CHAR
            if (not doesArmTouchGoals(arm.getEnd(), goals.copy())) and doesArmTouchObstacles(arm.getArmPos(), goals):
                 input_map[row][column] = WALL_CHAR
            # if (doesArmTouchGoals(arm.getEnd(), obstacles.copy())) :
            #     input_map[row][column] = WALL_CHAR
            if doesArmTouchGoals(arm.getEnd(), goals):
                input_map[row][column] = OBJECTIVE_CHAR
            # if doesArmTouchGoals(arm.getEnd(), goals) and input_map[row][column] != WALL_CHAR:
            #     input_map[row][column] = OBJECTIVE_CHAR


            # if doesArmTouchObstacles(arm.getArmPos(), obstacles.copy()):
            #     input_map[row][column] = WALL_CHAR
            # elif (doesArmTouchGoals(arm.getEnd(), obstacles.copy())) :
            #     input_map[row][column] = WALL_CHAR
            # elif doesArmTouchGoals(arm.getEnd(), goals.copy()):
            #     input_map[row][column] = OBJECTIVE_CHAR
            # elif doesarmpassthroughobstacles(arm.getArmPos(), obstacles.copy()):
            #     input_map[row][column] = WALL_CHAR
            # if not isArmWithinWindow(arm.getArmPos(), window):
            #     input_map[row][column] = WALL_CHAR
    # for alpha in range(range1[0], range1[1]+1):
    #     for belta in range(range2[0], range2[1]+1):
    #         arm.setArmAngle((alpha,belta))
    #         cur_angle = (alpha,belta)
    #         cur_pos = angleToIdx(cur_angle, offsets, granularity)
    #         # current_row = math.floor((alpha - range1[0])/granularity) + 1
    #         # current_column = math.floor((belta - range2[0])/granularity) + 1
    #         if doesArmTouchObstacles(arm.getArmPos(), obstacles.copy()):
    #             input_map[cur_pos[0]][cur_pos[1]] = WALL_CHAR
    #         if (doesArmTouchGoals(arm.getEnd(), obstacles.copy())) :
    #             input_map[cur_pos[0]][cur_pos[1]] = WALL_CHAR
    #         if doesArmTouchGoals(arm.getEnd(), goals.copy()):
    #             input_map[cur_pos[0]][cur_pos[1]] = OBJECTIVE_CHAR
    #         if not isArmWithinWindow(arm.getArmPos(), window):
    #             input_map[cur_pos[0]][cur_pos[1]] = WALL_CHAR
    # offsets = [range1[0], range2[0]]
    mazeinstance = Maze(input_map,offsets, granularity )

    # print(arm.getArmLimit())

    return mazeinstance
