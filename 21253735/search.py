# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
import collections
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod, [])(maze)

def pathreverse (pre, start, end):
    path = []
    h = end
    #change pre(h) to h
    # count = 0
    path.append(h)
    while (pre[h]!= start):
        h = pre[h]
        path.append(h)
    path.append(start)
    path.reverse()
    return path

# def bfs(maze):
#do we need to find the nearest goal from the start point
def bfs(maze):


    frontier = collections.deque()
    explored = set()
    path = []
    state = 0
    pre = {}
    start = maze.getStart()
    # print(start)
    end = maze.getObjectives()
    frontier.append(maze.getStart())
    # print(frontier)
    explored.add(maze.getStart())
    while len(frontier) > 0:
        current = frontier.popleft()
        state = state + 1
        if current in end:
            # print((pathreverse(pre, start, current)))
            # print(current)
            # print(pathreverse(pre,start,current))
            # print(len(pathreverse(pre,start,current)))
            return pathreverse(pre, start, current), state
        # elif current in explored:
        #     continue
        # state = state + 1
        for neighbors in maze.getNeighbors(*current):
            if neighbors not in explored:
                explored.add(neighbors)
                frontier.append(neighbors)
                pre[neighbors] = current
    # return []
    # TODO: Write your code here
    return [], 0

def dfs(maze):
    # TODO: Write your code here
    return [], 0

def greedy(maze):
    # TODO: Write your code here
    return [], 0

def astar(maze):
    # TODO: Write your code here
    return [], 0
