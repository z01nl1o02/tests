#
#https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
#
import os,sys,pdb
import numpy as np
import collections
import math
class GRAPH(object):
    def __init__(self,num,names=None):
        self.adjacent = np.zeros((num,num),np.int64)
        self.capacity = np.zeros((num,num),np.int64)
        self.flow = np.zeros((num,num),np.int64)
        self.back = np.zeros((num,num),np.int64)
        self.names = names
        return 
    def add_edge(self,s,t,capacity,flow = 0, back = 0):
        self.adjacent[s,t] = 1
        self.capacity[s,t] = capacity
        self.flow[s,t] = flow
        self.back[s,t] = back

        self.adjacent[t,s] = 1
        self.capacity[t,s] = capacity
        self.flow[t,s] = capacity - flow #revers flow
        self.back[t,s] = back
        return
    def __iter__(self):
        return iter(range(self.adjacent.shape[0]))
    def get_nbrs(self,s):
        nbrs = []
        for t,k in enumerate( self.adjacent[s,:].tolist() ):
            if k != 1:
                continue
            nbrs.append(t)
        return nbrs
    def get_capacity(self,s,t):
        return self.capacity[s,t]
    def get_flow(self,s,t):
        return self.flow[s,t]
    def inc_flow(self,s,t,delta):
        self.flow[s,t] += delta
        self.flow[s,t] = max( 0,self.flow[s,t]  )
        return
    def get_back(self,s):
        backs = []
        for b,k in enumerate( self.back[s,:].tolist() ):
            if k != 1:
                continue
            backs.append(b)
        return backs
    def BFS(self,S,T):
        deque = collections.deque()
        deque.append(S)
        mask = [False] * self.adjacent.shape[0]
        self.back = np.zeros( self.adjacent.shape, np.int64)
        while deque and self.back[T,:].sum() == 0:
            s = deque.popleft()
            for t in self.get_nbrs(s):
                if mask[t] == True:
                    continue
                if self.get_capacity(s,t) <= self.get_flow(s,t):
                    continue
                if t == S:
                    continue
                self.back[t,s] = 1
                mask[t] = True
                deque.append(t)
        return

    def EdmondsKarp(self,S,T):
        while 1:
            self.BFS(S,T)
            if self.back[T,:].sum() == 0:
                break #no more augemting path
            augmenting_path = []
            t = T
            augmenting_path.append(t)
            flows = []
            while t != S:
                prev = self.get_back(t)[0]
                augmenting_path.append(prev)
                flows.append( self.get_capacity(prev,t) - self.get_flow(prev,t) )
                t = prev

            flow = reduce(np.minimum, flows)
            augmenting_path.reverse()
            if self.names is None:
                print augmenting_path,flow
            else:
                tmp = [self.names[k] for k in augmenting_path]
                print tmp,flow
            t = T
            while t != S:
                prev = self.get_back(t)[0]
                self.inc_flow(prev,t,flow)
                self.inc_flow(t,prev,-1*flow)
                t = prev


def test():
    #A,B,C,D,E,F,G
    #0,1,2,3,4,5,6
    
    g = GRAPH(num=7,names='A,B,C,D,E,F,G'.split(','))
    g.add_edge(0,1,3)
    g.add_edge(0,3,3)
    g.add_edge(1,2,4)
    g.add_edge(2,0,3)
    g.add_edge(2,3,1)
    g.add_edge(2,4,2)
    g.add_edge(3,4,2)
    g.add_edge(3,5,6)
    g.add_edge(4,6,1)
    g.add_edge(4,1,1)
    g.add_edge(5,6,9)
    g.EdmondsKarp(0,6)
if __name__=="__main__":
    test()
