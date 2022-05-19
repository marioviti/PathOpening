import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def orientations3D(size=3):
    xx,yy,zz = np.meshgrid( range(-(size//2),size//2+1), range(-(size//2),size//2+1), range(-(size//2),size//2+1), indexing='ij')
    kernel_11 = np.logical_and(np.logical_and(xx>=0, zz>=0), yy>=0).astype(np.float32)
    kernel_12 = np.logical_and(np.logical_and(-xx>=0, -zz>=0), -yy>=0).astype(np.float32)
    kernel_21 = np.logical_and(np.logical_and(-xx>=0, zz>=0), yy>=0).astype(np.float32)
    kernel_22 = np.logical_and(np.logical_and(xx>=0, -zz>=0), -yy>=0).astype(np.float32)
    kernel_31 = np.logical_and(np.logical_and(-xx>=0, -zz>=0), yy>=0).astype(np.float32)
    kernel_32 = np.logical_and(np.logical_and(xx>=0, zz>=0), -yy>=0).astype(np.float32)
    kernel_41 = np.logical_and(np.logical_and(xx>=0, -zz>=0), yy>=0).astype(np.float32)
    kernel_42 = np.logical_and(np.logical_and(-xx>=0, zz>=0), -yy>=0).astype(np.float32)
    kernel_51 = (xx==1).astype(np.float32)
    kernel_52 = (-xx==1).astype(np.float32)
    kernel_61 = (yy==1).astype(np.float32)
    kernel_62 = (-yy==1).astype(np.float32)
    kernel_71 = (zz==1).astype(np.float32)
    kernel_72 = (-zz==1).astype(np.float32)
    kernal_all = np.stack([kernel_11,kernel_12,kernel_21,kernel_22,kernel_31,kernel_32,kernel_41,kernel_42,kernel_51,kernel_52,kernel_61,kernel_62,kernel_71,kernel_72])
    kernal_all[:,size//2,size//2,size//2] = 1
    return kernal_all

def orientations2D(size=3):
    xx,yy = np.meshgrid( range(-(size//2),size//2+1), range(-(size//2),size//2+1), indexing='ij')
    kernel_11 = np.logical_and(xx>=0, yy>=0).astype(np.float32)
    kernel_12 = np.logical_and(-xx>=0, -yy>=0).astype(np.float32)
    kernel_21 = np.logical_and(-xx>=0, yy>=0).astype(np.float32)
    kernel_22 = np.logical_and(xx>=0, -yy>=0).astype(np.float32)
    kernel_31 = (xx==1).astype(np.float32)
    kernel_32 = (-xx==1).astype(np.float32)
    kernel_41 = (yy==1).astype(np.float32)
    kernel_42 = (-yy==1).astype(np.float32)
    kernal_all = np.stack([kernel_11,kernel_12,kernel_21,kernel_22,kernel_31,kernel_32,kernel_41,kernel_42])
    kernal_all[:,size//2,size//2] = 1
    return kernal_all


class Convo(nn.Module):
    def __init__(self, kernel, input_channels, stride=1, padding=1, dilation=1, requires_grad=False):
        """
        for 1D kernel size must be Nfilters x Support
        for 2D kernel size must be Nfilters x Support_dim0 x Support_dim1
        for 3D kernel size must be Nfilters x Support_dim0 x Support_dim1 x Support_dim2
        """
        super(Convo, self).__init__()
        conv_funs = { 1: F.conv1d, 2: F.conv2d, 3: F.conv3d }
        self.convfun = conv_funs[len(kernel.shape)-1]
        kernel = kernel.unsqueeze(1).float()
        # THIS IS SO IMPORTANT ....
        self.kernel_tensor = th.nn.Parameter(kernel)
        self.kernel_tensor.requires_grad = requires_grad
        self.input_channels = input_channels
        self.dilation, self.stride, self.padding = dilation, stride, padding
        
    def forward(self, x):
        x = self.convfun(x, self.kernel_tensor, groups=int(self.input_channels), 
                         stride=self.stride, padding=self.padding, dilation=self.dilation)
        return x


def dilation3d(X, conn=26):
    assert conn in [6,26] , "3d connectivity only 8 or 26"
    if conn == 26:
        return  F.max_pool3d(X, 3, 1, 1)
    if conn == 6:
        p1 = F.max_pool3d(X, (3,1,1), (1,1,1), (1,0,0))
        p2 = F.max_pool3d(X, (1,3,1), (1,1,1), (0,1,0))
        p3 = F.max_pool3d(X, (1,1,3), (1,1,1), (0,0,1))
        return th.max(th.max(p1,p2),p3)
    
def erosion3d(X, conn=26):
    return -dilation3d(-X, conn=conn)

def opening3d(X, conn_e=6, conn_d=26):
    return dilation3d(erosion3d(X, conn=conn_e), conn=conn_d)

def closing3d(X, conn_e=26, conn_d=6):
    return erosion3d(dilation3d(X, conn=conn_d), conn=conn_e)


def dilation2d(X, conn=8):
    assert conn in [4,8] , "2d connectivity only 4 or 8"
    if conn == 8:
        return  F.max_pool2d(X, 3, 1, 1)
    if conn == 4:
        p1 = F.max_pool2d(X, (3,1), (1,1), (1,0))
        p2 = F.max_pool2d(X, (1,3), (1,1), (0,1))
        return th.max(p1,p2)
    
def erosion2d(X, conn=8):
    return -dilation2d(-X, conn=conn)

def opening2d(X, conn_e=4, conn_d=8):
    return dilation2d(erosion2d(X, conn=conn_e), conn=conn_d)

def closing2d(X, conn_e=8, conn_d=4):
    return erosion2d(dilation2d(X, conn=conn_d), conn=conn_e)
 
def path_opening2D(I, n_iter, size=3, robustify=False):
    kernel = orientations2D(size=size)
    tkernel = th.from_numpy(kernel)
    gc = Convo(tkernel, 1, padding=size//2)
    gc.to(I.device)
    oriI = F.relu((gc(I)).clamp_(0,2)*I - I)
    accI = I.expand_as(oriI) + oriI
    gc = Convo(tkernel, tkernel.shape[0], padding=size//2)
    gc.to(I.device)
    for i in range(n_iter):
        oriI = F.relu(gc(oriI).clamp_(0,2)*I - I)
        if robustify:
            oriI = closing2d(oriI)
        accI += oriI
    b,c,h,w = accI.shape
    oriIsum = th.zeros([b,c//2,h,w])
    for i in range(0,c-1,2):
        oriIsum[:,i//2] = F.relu( accI[:,i] + accI[:,i+1] - 1.0 )
    openI, _ = oriIsum.max(axis=1, keepdims=True)
    return openI, oriIsum

def path_opening3D(I, n_iter, size=3):
    # initial orientation (base)
    kernel = orientations3D(size=size)
    tkernel = th.from_numpy(kernel)
    c = I.shape[1]
    gc = Convo(tkernel, c, padding=size//2)
    gc.to(I.device)
    oriI = F.relu((gc(I)).clamp_(0,2)*I - I)
    accI = I.expand_as(oriI) + oriI
    gc = Convo(tkernel, tkernel.shape[0], padding=size//2)
    gc.to(I.device)
    for i in range(n_iter):
        oriI = F.relu(gc(oriI).clamp_(0,2)*I - I)
        accI += oriI
    b,c,h,w,d = accI.shape
    oriIsum = th.zeros([b,c//2,h,w,d])
    for i in range(0,c-1,2):
        oriIsum[:,i//2] = F.relu( accI[:,i] + accI[:,i+1] - 1.0 )
    openI, _ = oriIsum.max(axis=1, keepdims=True)
    return openI, oriIsum
    
    
def rorpo2D(I, n_iter, size=3):
    openI, oriIsum = path_opening2D(I, n_iter, size=size)
    rorpo = th.max(oriIsum, axis=1, keepdims=True)[0] - th.median(oriIsum, axis=1, keepdims=True)[0]
    return rorpo

def rorpo3D(I, n_iter, size=3):
    openI, oriIsum = path_opening3D(I, n_iter, size=size)
    A1,A4 = th.max(oriIsum, axis=1, keepdims=True)[0], th.median(oriIsum, axis=1, keepdims=True)[0]
    rorpo = (A1-A4)
    return rorpo

def minmaxnorm(I):
    """
    I shape: b,c,*spatial e.g. (h,w,d)
    """
    old_shape = I.shape
    view_I = I.view(*I.shape[:2],np.prod(I.shape[2:]))
    min_v, max_v = view_I.min(axis=-1,keepdims=True)[0], view_I.max(axis=-1,keepdims=True)[0]
    mmI = (view_I - min_v) / (max_v - min_v)
    mmI = mmI.reshape(*old_shape)
    return mmI
                                                           
