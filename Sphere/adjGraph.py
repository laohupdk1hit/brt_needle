#
#  adjGraph
#
#  Created by Brad Miller on 2005-02-24.
#  Copyright (c) 2005 Brad Miller, David Ranum, Luther College. All rights reserved.
#

import sys
import os
import unittest

from sqlalchemy import true
from Sphere.deque import Deque

class Graph:
    def __init__(self):
        self.vertices = {}
        self.numVertices = 0
        self.numEdge = 0
        self.numtriangles = 0
        self.triangles = Deque()

    def initcolor(self):
        for v in self.vertices.keys():
            self.vertices[v].color = 0
    
    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertices[key] = newVertex
        return newVertex
    
    def getVertex(self,n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertices
    
    def addDoubleEdge(self,f,t,cost=0):
        if f not in self.vertices:
            nv = self.addVertex(f)
        if t not in self.vertices:
            nv = self.addVertex(t)

        self.vertices[f].addNeighbor(self.vertices[t],cost)
        self.vertices[t].addNeighbor(self.vertices[f],cost) #w无向图，互相添加
        self.numEdge += 1

    def addSingleEdge(self,f,t,cost=0):
        if f not in self.vertices:
            nv = self.addVertex(f)
        if t not in self.vertices:
            nv = self.addVertex(t)
        self.vertices[f].addNeighbor(self.vertices[t],cost)


    def delEdge(self,f,t):
        self.vertices[f.id].delNeighbor(t)
        self.vertices[t.id].delNeighbor(f) #w无向图，互相添加
        self.numEdge -= 1
        
       
    def addTriangle(self,a,b,c):
        ai=a.id
        bi=b.id
        ci=c.id
        if ai not in self.vertices:
            nv = self.addVertex(a)
        if bi not in self.vertices:
            nv = self.addVertex(b)
        if ci not in self.vertices:
            nv = self.addVertex(c)

        if not((a.isCto(b)) and (a.isCto(c)) and (b.isCto(c))):
            self.numtriangles += 1

        if not(a.isCto(b)):
            self.addDoubleEdge(ai,bi)
        if not(a.isCto(c)):
            self.addDoubleEdge(ai,ci)
        if not(b.isCto(c)):
            self.addDoubleEdge(bi,ci)
        self.triangles.addFront([ai,bi,ci])


    def getVertices(self):
        return list(self.vertices.keys())
        
    def __iter__(self):
        return iter(self.vertices.values())
                
class Vertex:
    def __init__(self,num):
        self.id = num
        self.connectedTo = {}
        self.color = 0 #用color和child去除重复，color标记是否已经访问过
        self.child = []
        self.position = [0,0,0]

    # def __lt__(self,o):
    #     return self.id < o.id
    
    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr.id] = weight
    
    def delNeighbor(self,nbr):
        if (nbr.id in self.connectedTo.keys()):
            self.connectedTo.pop(nbr.id)
        
    def setChild(self,pa,c):
        self.child.append([pa,c]) #parent在前
        

    def setPosition(self,pos):
        self.position = pos

    def isCto(self,t):
        return t.id in self.connectedTo.keys()


    def getPosition(self):
        return self.position
        
    def getChild(self,pa):
        for i in range(len(self.child)):
            if pa.id == self.child[i][0].id:
                ch = self.child[i][1]
                break
            else:
                ch = []
        return ch

    def getColor(self):
        return self.color
    
    def getConnections(self):
        return self.connectedTo.keys()
        
    def getWeight(self,nbr):
        return self.connectedTo[nbr]
                
    def __str__(self):
        #return str(self.id) + ":color " + self.color +  ":pred \n\t[" + str(self.pred)+ "]\n"
        return str(self.id)
    
    def getId(self):
        return self.id

class adjGraphTests(unittest.TestCase):
    def setUp(self):
        self.tGraph = Graph()
        
    def testMakeGraph(self):
        gFile = open("test.dat")
        for line in gFile:
            fVertex, tVertex = line.split('|')
            fVertex = int(fVertex)
            tVertex = int(tVertex)
            self.tGraph.addEdge(fVertex,tVertex)
        for i in self.tGraph:
            adj = i.getAdj()
            for k in adj:
                print(i, k)

        
if __name__ == '__main__':
    unittest.main()
              
