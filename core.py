# MIT License

# Copyright (c) 2021 Nguyen Do Quoc Anh (Cemu0)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import math
import numpy as np
import torch.nn as nn
device = torch.device('cuda')  #'cpu'

class cell(nn.Module):
    def __init__(self, input_sz:int, hidden_sz:int, output_sz:int):
        super().__init__()#same as the nn.Module obj
        self.weights1 = nn.Parameter(torch.randn(input_sz, hidden_sz) / math.sqrt(input_sz),requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(hidden_sz),requires_grad=True)
        self.weights2 = nn.Parameter(torch.randn(hidden_sz, output_sz) / math.sqrt(hidden_sz),requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(output_sz),requires_grad=True)
    def forward(self,Input):
        a = nn.ReLU()(Input.clone() @ self.weights1 + self.bias1) #.clone()
        a = nn.LogSigmoid()(a @ self.weights2 + self.bias2) #ReLU Sigmoid ReLU6 Tanh log_softmax .clone()  LogSigmoid
        return a

# algorithm and code by Cemu0 
class NNN:                                                
    def __init__ (self,
                  InputSize:int,
                  OutputSize:int,           
                  HiddenLayerPerCell = [], # hidden layer per cell 
                  Connections = [], #2D rectagele matrix for connections bettwen cell
                  ActivateHoocmon = 0.5): # threadhold 
        
        '''define requet data'''  # ✔
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        self.Cnts = Connections
        self.cells = len(self.Cnts) # cells >= 2
        # self.ActivateHoocmon = ActivateHoocmn
        self.CellList = []
        
        '''generating net'''
        # add the first cell ✔  #collum 0
        print("input:",self.InputSize+sum(self.Cnts[:,0]))
        self.CellList.append(cell(self.InputSize+sum(self.Cnts[:,0]),\
                                 HiddenLayerPerCell[0],\
                                 sum(self.Cnts[0,:])).to(device)) #rown 0
        # add orther cells ✔
        for i in range(self.cells-2):
            AllConnectsToThisCell = sum(self.Cnts[:,i+1])
            OutputOfThisCell = sum(self.Cnts[i+1]) 
            self.CellList.append(cell(AllConnectsToThisCell,\
                                      HiddenLayerPerCell[i+1],\
                                      OutputOfThisCell).to(device))
        # add the last cell ✔
        self.CellList.append(cell(sum(self.Cnts[:,self.cells-1]),\
                                 HiddenLayerPerCell[self.cells-1],\
                                 sum(self.Cnts[self.cells-1])+self.OutputSize).to(device)) #!!!
        
    def Forward(self,xb,yb,step,cal = False):
        """ define variable """
        # passing data into cell[0] ... actually is cell 1
        # Additional Input Required
        air = sum(self.Cnts[:,0])
        # this must work for the truth
        x_input = F.pad(xb,(0,air)) # add input place
        InputList = [] #less than 1
        LastInputList = []
        OutputList = []
        y_predict = None
        
        for cell in range(self.cells):
            #input of all cell
            AllConnectsToThisCell = sum(self.Cnts[:,cell])
            InputList.append(torch.zeros(xb.shape[0],AllConnectsToThisCell,device=device))
            
            #output of all cell except last cell
            if cell != self.cells-1:
                OutputOfThisCell = sum(self.Cnts[cell])
                OutputList.append(torch.zeros(xb.shape[0],OutputOfThisCell,device=device))
            else:
                #output of the last cell
                OutputList.append(torch.zeros(xb.shape[0],sum(self.Cnts[self.cells-1])+self.OutputSize,device=device))
            
            
        """ caculate """
        # prepare the net 
#       with torch.no_grad():
        self.CellList[0].forward(x_input)
        # let the net .. think by forward throw every cells
        # the result of cell 
        for thingS in range(step):
            LastOutputList = OutputList.copy()
            for i in range(self.cells): 
                    InputList[cell][:,sum(self.Cnts[:,cell][:i]):sum(self.Cnts[:,cell][:i+1])]\
                        = LastOutputList[i][:,sum(self.Cnts[i][:cell]):sum(self.Cnts[i][:cell+1])].clone()
            for cell in range(self.cells):
                if cell != 0:
                    OutputList[cell] = self.CellList[cell].forward(InputList[cell])
                else:
                    # first cell
                    x_input[:,:sum(self.Cnts[:,0])] = InputList[cell]#T
                    OutputList[cell] = self.CellList[0].forward(x_input)
            y_predict = OutputList[self.cells-1][:,:self.OutputSize]
            if cal: # check if the loss is lower
                loss = loss_func(y_predict, yb)
                print("inside loss", loss)
                
        return y_predict
