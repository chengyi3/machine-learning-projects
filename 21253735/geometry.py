# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *
# from armlink import ArmLink
# from arm import Arm
# import copy
# from arm import Arm

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise


        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    xcoordinate = start[0]
    ycoordinate = start[1]
    angle = math.radians(angle)
    endx = xcoordinate + length* math.cos(angle)
    endy = ycoordinate - length* math.sin(angle)

    return (endx, endy)

def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    # I have tried several algorithms on the internet, this algorithm comes from
    # https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    for armpos in armPos:
        for obstacle in obstacles:
            startpoint = armpos[0]
            endpoint = armpos[1]
            x = obstacle[0]
            y = obstacle[1]
            r = obstacle[2]
            startx = startpoint[0]
            starty = startpoint[1]
            endx = endpoint[0]
            endy = endpoint[1]
            disa =  math.pow(x-startx,2)+math.pow(y-starty,2)
            disb =  math.pow(endy-y,2)+math.pow(endx - x,2)
            s = np.array(startpoint)
            e = np.array(endpoint)
            o = np.array((x,y))
            dis = np.linalg.norm(np.cross(e-s, s-o))/np.linalg.norm(e-s)
            unitvector = (e-s)/np.linalg.norm(e-s)
            min_endpointdistance = min(math.sqrt(disa), math.sqrt(disb))
            j = unitvector[0]*(o[0] - s[0]) + unitvector[1]*(o[1] - s[1])
            px = j * unitvector[0] + s[0]
            py = j * unitvector[1] + s[1]
            if (px >= s[0] and px <= e[0]) or (px >= e[0] and px <= s[0]):
                if (py >= e[1] and py <= s[1]) or (py >= s[1] and py <= e[1]):
                    # min_dis = dis
                    if (dis <= r):
                        return True
            else:
                if (min_endpointdistance <= r):
                    return True
    #using algorithm on https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    # for armpos in armPos:
    #     for obstacle in obstacles:
    #         startpoint = armpos[0]
    #         endpoint = armpos[1]
    #         x = obstacle[0]
    #         y = obstacle[1]
    #         r = obstacle[2]
    #         startx = startpoint[0]
    #         starty = startpoint[1]
    #         endx = endpoint[0]
    #         endy = endpoint[1]
    #         disa =  math.pow(x-startx,2)+math.pow(y-starty,2)
    #         disb =  math.pow(endy-y,2)+math.pow(endx - x,2)
    #         lsquare = math.pow(endy-starty,2) + math.pow(endx-startx,2)
    #         c = [x-startx, y-starty]
    #         d = [endx-startx, endy-starty]
    #         dotproduct = np.dot(c,d)
    #         f = max(0,min(1,dotproduct/lsquare))
    #         k = [startx + f*d[0] , starty + f*d[1] ]
    #         dissquare = math.pow(x-k[0],2)+math.pow(y-k[1],2)
    #         if (dissquare <= r*r or disa<= r*r or disb<=r*r):
    #             return True


    # for armpos in armPos:
    #     for obstacle in obstacles:
    #         startpoint = armpos[0]
    #         endpoint = armpos[1]
    #         x = obstacle[0]
    #         y = obstacle[1]
    #         r = obstacle[2]
    #         startx = startpoint[0]
    #         starty = startpoint[1]
    #         endx = endpoint[0]
    #         endy = endpoint[1]
    #         disa =  math.pow(x-startx,2)+math.pow(y-starty,2)
    #         disb =  math.pow(endy-y,2)+math.pow(endx - x,2)
    #         k = endx - startx
    #         m = endy - starty
    #         n = math.pow(k,2) + math.pow(m,2)
    #         l1 = (x-startx)*k
    #         l2 = (y-starty)*m
    #         l = (l1+l2)/float(n)
    #         if l > 1:
    #             l = 1
    #         if l < 0:
    #             l = 0
    #         t = startx + l*k
    #         s = starty + l*m
    #         d1 = t - x
    #         d2 = s - y
    #         dissquare = math.pow(d1,2)+math.pow(d2,2)
    #         if (math.sqrt(dissquare) <= r or math.sqrt(disa) <= r or math.sqrt(disb) <= r):
    #             return True



    return False

def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    for goal in goals:
        x = goal[0]
        y = goal[1]
        r = goal[2]
        dis = (math.pow(x-armEnd[0],2) + math.pow(y-armEnd[1],2))
        if (math.sqrt(dis) <= r):
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window
        Return:
            True if all parts are in the window. False it not.
    """
    for arm in armPos:
        start = arm[0]
        end = arm[1]
        if (start[0] > window[0] or start[1] > window[1] or end[0] > window[0] or end[1] > window[1]):
            return False
        if (start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0):
            return False
        # if (start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0) :
        #     return false

    return True
