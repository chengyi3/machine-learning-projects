# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
# from queue import PriorityQueue
import heapq
from heapq import heappop, heappush
import collections
# import pdb;
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "astar": astar,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)

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
def pathreverse2 (pre, start, end):
    path = []
    h = end

    # path.append(h)
    while (pre[h]!= start):
        h = pre[h]
        path.append(h)



    path.append(start)
    path.reverse()
    return path
def bfs(maze):

    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # frontier = []
    frontier = collections.deque()
    explored = set()
    path = []
    # step = 0
    pre = {}
    start = maze.getStart()
    end = maze.getObjectives()[0]
    frontier.append(maze.getStart())
    explored.add(maze.getStart())
    while len(frontier) > 0:
        current = frontier.popleft()
        # step++
        if current == end:
            return pathreverse(pre, start, end)
        for neighbors in maze.getNeighbors(*current):
            if neighbors not in explored:
                explored.add(neighbors)
                frontier.append(neighbors)
                pre[neighbors] = current
    return []


def dfs(maze):

    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    #change list to deque
    # frontier = []
    frontier = collections.deque()
    pre = {}
    explored = set()
    path = []
    start = maze.getStart()
    end = maze.getObjectives()[0]
    # step = 0
    frontier.append(start)
    explored.add(start)
    while len(frontier) > 0:
        current = frontier.pop()
        # step++
        if current == end:
            return pathreverse(pre, start, end)
        for neighbors in maze.getNeighbors(*current):
            if neighbors not in explored:
                explored.add(neighbors)
                frontier.append(neighbors)
                pre[neighbors] = current
    return []

def manhantanndistance(position1, position2):
    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])
# https://en.wikipedia.org/wiki/A*_search_algorithm
def astar(maze):

    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    end = maze.getObjectives()[0]
    frontier = []
    explored = set()
    pre = {}
    #gcost is cost from start to n
    gcost = {}
    fcost = {}
    fcost[start] = manhantanndistance(start, end)
    gcost[start] = 0
    frontier.append(start)
    #f = g + h
    #change PriorityQueue to heapq
    # q = PriorityQueue()
    q = []
    heappush(q,(manhantanndistance(start,end), start))
    # q.put(start, manhantanndistance(start, end))
    while len(frontier) > 0:
        # current = q.get()
        current = heappop(q)[1]
        if current == end :
            return pathreverse(pre, start, current)
        frontier.remove(current)
        explored.add(current)
        for neighbors in maze.getNeighbors(*current):
            if neighbors in explored:
                continue
            newgcost = gcost[current] + 1
            if neighbors not in gcost or newgcost < gcost[neighbors]:
                pre[neighbors] = current
                gcost[neighbors] = newgcost
                fcost[neighbors] = gcost[neighbors] + manhantanndistance(neighbors, end)
                if neighbors not in frontier:
                    frontier.append(neighbors)
                    # q.put(neighbors, fcost[neighbors])
                    heappush(q, (fcost[neighbors], neighbors))
    return []
#find the closest goal from the current position
# def astardis (position1, position2, maze):
#     start = position1
#     end = position2
#     frontier = []
#     explored = set()
#     pre = {}
#
#     gcost = {}
#     fcost = {}
#     fcost[start] = manhantanndistance(start, end)
#     gcost[start] = 0
#     frontier.append(start)
#     p = []
#     if position1 == position2:
#         return 0
#
#     q = PriorityQueue()
#     q.put(start, manhantanndistance(start, end))
#     while len(frontier) > 0:
#         current = q.get()
#         if current == end :
#             p = pathreverse(pre, position1, position2)
#             return len(p)
#         frontier.remove(current)
#         explored.add(current)
#         for neighbors in maze.getNeighbors(*current):
#             if neighbors in explored:
#                 continue
#             newgcost = gcost[current] + 1
#             if neighbors not in gcost or newgcost < gcost[neighbors]:
#                 pre[neighbors] = current
#                 gcost[neighbors] = newgcost
#                 fcost[neighbors] = gcost[neighbors] + manhantanndistance(neighbors, end)
#                 if neighbors not in frontier:
#                     frontier.append(neighbors)
#                     q.put(neighbors, fcost[neighbors])
#
#     return 0


def closestdistance(position, dots, visited):
    # que = PriorityQueue()
    que = []
    for dot in dots:
        if dot not in visited:
            # que.put(dot, manhantanndistance(position, dot))
            heappush(que,(manhantanndistance(position, dot), dot))

    # closestdot = que.get()
    closestdot = heappop(que)[1]
    return closestdot

def minimumspanningtree(position, unvisited):
    points = unvisited.copy()
    # print(points)
    points.append(position)

    current = list()
    current.append(position)
    start = position
    # print(start)
    if start in points:
        points.remove(start)

    cost = 0

    while(len(points) > 0):
        # best_point = PriorityQueue()
        best_point = []
        for point in points:
            for current_point in current:
                # best_point.put([point,manhantanndistance(current_point,point)],manhantanndistance(current_point,point))
                heappush(best_point, (manhantanndistance(current_point, point),point))
        # new_point = best_point.get()
        a = heappop(best_point)
        new_point = a[1]
        # points.remove(new_point[0])
        points.remove(new_point)
        # current.append(new_point[0])
        current.append(new_point)
        cost+=a[0]

    return cost
    # visited = []
    # dis = {}
    # d = 0
    # unexplored = unvisited
    # visited.append(position)
    # dis[position] = 0
    # q = PriorityQueue()
    # q.put(position, 0)
    # while len(visited) < len(unvisited) + 1 :
    #         # pdb.set_trace()
    #     current = q.get()
    #     if current in unexplored:
    #         unexplored.remove(current)
    #
    #     if current not in visited:
    #         visited.append(current)
    #     for dot in unexplored:
    #         if dot not in dis or manhantanndistance(current, dot) < dis[dot]:
    #             #print(type(dot))
    #             #print(type(current))
    #             dis[dot] = manhantanndistance(current, dot)
    #             q.put(dot, dis[dot])
    # for element in visited:
    #     # print(*element)
    #     d = d + dis[element]
    # return d
#    hcost = {}
#    h = 0
# # cost = {}
#    qu = PriorityQueue()
#    while len(unvisited) != 0:
#      for objective in visited:
#          for neighbor in unvisited:
#              if neighbor not in hcost or manhantanndistance(objective, neighbor) < hcost[neighbor]:
#                hcost[neighbor] = manhantanndistance(objective, neighbor)
#
#      mindis = min(hcost.values())
#      h = h + mindis
#      nearestdot = [key for key in hcost if hcost[key] == mindis]
#      visited.append(nearestdot[0])
#      hcost.pop(nearestdot[0], None)
#      unvisited.remove(nearestdot[0])
#    return  h
def findnearstdot(position, end) :
        mindis = 0
        nearestdot = position
        for dot in end:
            dis =  manhantanndistance(position, dot)
            if mindis == 0 or dis < mindis:
                mindis = dis
                nearestdot = dot
        return nearestdot

def astar_multi(maze):

    """
    Runs A star for part 2 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # start = maze.getStart()
    # end = maze.getObjectives()
    # #
    # paths = list()
    #
    # while(len(end) != 0):
    #     frontier = []
    #     explored = set()
    #     pre = {}
    #
    #     gcost = {}
    #     fcost = {}
    #     gcost[start] = 0
    #     frontier.append(start)
    #
    #     q = []
    #
    #
    #     # mindis = 0
    #     # nearestdot = start
    #     # for dot in end:
    #     #     dis =  manhantanndistance(start, dot)
    #     #     if mindis == 0 or dis < mindis:
    #     #         mindis = dis
    #     #         nearestdot = dot
    #     m = minimumspanningtree(start, end)
    #     fcost[start] =  m
    #     heappush(q,(m, start))
    #     while len(q) > 0:
    #
    #         current = heappop(q)[1]
    #
    #         if(current in end):
    #             end.remove(current)
    #             if (len(end) == 0) :
    #                 apath = pathreverse(pre,start,current)
    #
    #             else:
    #                 apath = pathreverse2(pre,start,current)
    #             paths.append(apath)
    #
    #             start = current
    #             break
    #
    #         # q.remove(current)
    #         explored.add(current)
    #
    #
    #
    #         for neighbors in maze.getNeighbors(*current):
    #             if neighbors in explored:
    #                 continue
    #             newgcost = gcost[current] + 1
    #             if neighbors not in gcost or newgcost < gcost[neighbors]:
    #                 pre[neighbors] = current
    #                 gcost[neighbors] = newgcost
    #
    #                 dot = findnearstdot(neighbors, end)
    #                 fcost[neighbors] = gcost[neighbors]+ manhantanndistance(neighbors, dot) + minimumspanningtree(dot, end)
    #                 if neighbors not in q:
    #                     # .append(neighbors)
    #
    #                     heappush(q,(fcost[neighbors], neighbors))
    #
    # finalpath = []
    # for path in paths:
    #
    #
    #     finalpath.extend(path)
    # # print(finalpath)
    # return finalpath

    #
    start = maze.getStart()
    end = maze.getObjectives()

    paths = list()

    while(len(end) != 0):
        frontier = []
        explored = set()
        pre = {}

        gcost = {}
        fcost = {}
        fcost[start] = minimumspanningtree(start, end)
        gcost[start] = 0
        frontier.append(start)

        q = []

        heappush(q,(fcost[start], start))
        while len(q) > 0:

            current = heappop(q)[1]

            if(current in end):
                end.remove(current)
                # if (len(end) == 0) :
                #     apath = pathreverse(pre,start,current)
                #
                # else:
                apath = pathreverse2(pre,start,current)
                # apath = pathreverse(pre,start,current)
                paths.append(apath)

                start = current
                break

            # frontier.remove(current)
            explored.add(current)



            for neighbors in maze.getNeighbors(*current):
                if neighbors in explored:
                    continue
                newgcost = gcost[current] + 1
                if neighbors not in gcost or newgcost < gcost[neighbors]:
                    pre[neighbors] = current
                    gcost[neighbors] = newgcost


                    fcost[neighbors] = gcost[neighbors] + minimumspanningtree(neighbors, end)
                    if neighbors not in q:
                        # frontier.append(neighbors)

                        heappush(q,(fcost[neighbors], neighbors))

    finalpath = []
    for path in paths:


        finalpath.extend(path)
    # print(finalpath)
    finalpath.append(start)
    # print(finalpath)
    return finalpath

def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    dots = []
    pre = {}
    start = maze.getStart()
    dots = maze.getObjectives()
    n = len(dots)
    visited = []
    visited.append(start)
    m = 0
    p = []

    # gcost[start] = 0
    while len(visited) != n + 1:
            nextdot = closestdistance(visited[m], dots, visited)
            star = visited[m]
            end = nextdot
            frontier = []
            explored = set()
            pr = {}
            #gcost is cost from start to n
            gcost = {}
            fcost = {}
            fcost[star] = manhantanndistance(star, end)
            gcost[star] = 0
            frontier.append(star)
            #f = g + h
            # q = PriorityQueue()
            q = []
            heappush(q,(manhantanndistance(star, end), star))
            # q.put(star, manhantanndistance(star, end))
            while len(frontier) > 0:
                # current = q.get()
                current = heappop(q)[1]
                if current == end :
                    p.extend(pathreverse2(pr, star, current))
                    break
                frontier.remove(current)
                explored.add(current)
                for neighbors in maze.getNeighbors(*current):
                    if neighbors in explored:
                        continue
                    newgcost = gcost[current] + 1
                    if neighbors not in gcost or newgcost < gcost[neighbors]:
                        pr[neighbors] = current
                        gcost[neighbors] = newgcost
                        fcost[neighbors] = gcost[neighbors] + manhantanndistance(neighbors, end)
                        if neighbors not in frontier:
                            frontier.append(neighbors)
                            # q.put(neighbors, fcost[neighbors])
                            heappush(q,(fcost[neighbors], neighbors))
             # p = []
        # pre[nextdot] = visited[m]
            visited.append(nextdot)
            m = m+1
        # if len(visited) == n:

        #     end = visited[m]
    p.append(visited.pop())
    # print(p)
    return p
