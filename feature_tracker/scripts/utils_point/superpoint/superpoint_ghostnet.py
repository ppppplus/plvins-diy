import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
from time import time

from utils_point.superpoint.ghostnet_modules.g_ghost_backbone import GGhost_Backbone
from utils_point.superpoint.ghostnet_modules.cnn_heads import DetectorHead, DescriptorHead
from utils.base_model import BaseExtractModel

class SuperPoint_GGhost(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, weight_path):
        super(SuperPoint_GGhost, self).__init__()
        # print(" Running SuperPoint:",model_name="")
        self.backbone = GGhost_Backbone("032")
        self.detector_head = DetectorHead(input_channel=128,
                                          grid_size=8, using_bn=True)
        self.descriptor_head = DescriptorHead(input_channel=128,
                                              output_channel=128,
                                              grid_size=8, using_bn=True)
        checkpoint = torch.load(weight_path)
        # checkpoint = torch.load(model_path,map_location=torch.device('cpu'))        
        self.load_state_dict(checkpoint)#）

    def forward(self, x):
        feat_map = self.backbone(x)
        scores = self.detector_head(feat_map)
        descriptors = self.descriptor_head(feat_map)
        return scores,descriptors
    
class SuperpointPointGhostNetExtractModel(BaseExtractModel):
    def _init(self, params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.params = params
        self.cell = 8
        self.border_remove = 4
        self.conf_thresh=self.params["conf_thresh"]
        self.nms_dist = self.params["nms_dist"]
        self.spghostnet = SuperPoint_GGhost(self.params["weight_path"])  
        self.spghostnet = self.spghostnet.eval()
        self.spghostnet = self.spghostnet.to(self.device)
        

    def process_image(self, img):
        """ convert image to grayscale and resize to img_size.
        Inputs
        impath: Path to input image.
        img_size: (W, H) tuple specifying resize size.
        Returns
        grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        
        # Image is resized via opencv.
        # interp = cv2.INTER_AREA
        # grayim = cv2.resize(grayim, (self.params['W'], self.params['H']), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)[None, None]
        return grayim, True

    def extract(self, img):
        # input img and output feature points   
        # This class runs the SuperPoint network and processes its outputs.
        grayim, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        grayim = torch.from_numpy(grayim).to(self.device)
        self.H, self.W = img.shape[0], img.shape[1]
        # Get points and descriptors.
        with torch.no_grad():
            semi, coarse_desc = self.spghostnet.forward(grayim)
        # print("superpoint run time:%f", end_time-start_time)
        # feature_points = pts[0:2].T.astype('int32')
        start_time = time()
        pts, desc = self.process_output(semi, coarse_desc)
        end_time = time()
        print("output process time:{}".format(end_time-start_time))
        # print("++++", pts.shape, desc.shape)
        return pts, desc # [3,num_points], [256,num_points]
  
    def process_output(self, semi, coarse_desc):
        semi = semi.squeeze()
        dense = torch.exp(semi) # Softmax.
        dense = dense / (torch.sum(dense, axis=0)+.00001) # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(self.H / self.cell)
        Wc = int(self.W / self.cell)
        nodust = nodust.permute(1, 2, 0)
        heatmap = torch.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = heatmap.permute(0, 2, 1, 3)
        heatmap = torch.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
        xs, ys = torch.where(heatmap >= self.conf_thresh) # Confidence threshold.
        pts = torch.zeros((3, len(xs))) # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        # pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, self.H, self.W, dist_thresh=self.nms_dist) # Apply NMS.
        inds = torch.argsort(pts[2,:])
        inds = torch.flip(inds, dims=[0])
        pts = pts[:,inds] # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = torch.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W-bord))
        toremoveH = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H-bord))
        toremove = torch.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = torch.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = pts[:2, :]
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.H)/2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            # desc = desc.reshape(D, -1)
            # desc = torch.norm(desc)[None]
        return pts.cpu().numpy(), desc
    
    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = torch.zeros((H, W), dtype=int) # Track NMS data.
        inds = torch.zeros((H, W), dtype=int) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = torch.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().int() # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return torch.zeros((3,0)).int(), torch.zeros(0).int()
        if rcorners.shape[1] == 1:
            out = torch.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, torch.zeros((1), dtype=int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = torch.nn.functional.pad(grid, (pad,pad,pad,pad), mode='constant', value=0)
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = torch.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = torch.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds