from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import cv2
import os
from imutils.video import VideoStream
import time
import numpy as np
import torch
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pspnet import PSPNet
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
from utils.metrics import MultiThresholdMeasures
from tkcolorpicker import askcolor
import colorsys

y = np.zeros(3)
active = 0
def str2bool(s):
    return s.lower() in ('t', 'true', 1)

def change_color(image):
    hsl_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    hsl_image[:,:,0] = y[0]*180
    # hsl_image[:,:,1] = y[1]
    hsl_image[:,:,2] = y[2]*255
    new_img = cv2.cvtColor(hsl_image,cv2.COLOR_HLS2RGB)
    return new_img

class PBA:
    
    
    def __init__(self,vs,args):
           
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
        
        self.root = tki.Tk()
        self.panel = None
        
        btn1 = tki.Button(self.root, text="Pick",command=self.colorpick)
        btn1.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
        btn2 = tki.Button(self.root, text="Default",command=self.setinactive)
        btn2.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
        
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args = (args,))
        self.thread.start()
        
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
    def videoLoop(self,args):
        train_model = './pspnet_resnet101_trained.pth'
        device = 'cuda' if args["use_gpu"] else 'cpu'
    
        net = PSPNet().to(device)
        state = torch.load(train_model, map_location=lambda storage, loc: storage)
        net.load_state_dict(state['weight'])
    
        test_joint_transforms = jnt_trnsf.Compose([
            jnt_trnsf.Safe32Padding()
            ])
    
        test_image_transforms = std_trnsf.Compose([
            std_trnsf.ToTensor(),
            std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
        mask_transforms = std_trnsf.Compose([
            std_trnsf.ToTensor()
        ])
    
        
    
        metric = MultiThresholdMeasures()
        metric.reset()
        durations = list()
        
        
        try:
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                data = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
                image_n = data
                data = Image.fromarray(data)
                iw,ih = data.size   
                data = test_joint_transforms(data,None)
                data = test_image_transforms(data)
                data = data.unsqueeze(0)
                
                with torch.no_grad():
                    if(active):
                        net.eval()
                        data = data.to(device)
                        start = time.time()
                        logit = net(data)
                        duration = time.time() - start
                        pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
                        mh, mw = data.size(2), data.size(3)
                        mask = pred >= 0.5
                        mask_n = np.zeros((mh, mw, 3))
                        mask_n[:,:,0] = mask
                        mask_n[:,:,1] = mask
                        mask_n[:,:,2] = mask
                        mask_n = np.uint8(mask_n)
                        # print(image_n.shape)
                        # print(mask_n.shape)
                        x = np.zeros((mh,mw,3))
                        x[:,:,0] = image_n[:,:,0]*mask
                        x[:,:,1] = image_n[:,:,1]*mask
                        x[:,:,2] = image_n[:,:,2]*mask
                        # x = cv2.bitwise_and(image_n, mask_n, None)
                        x = np.uint8(x)
                        image_n = image_n - x + change_color(x)
                        # mask_n[:,:,0] = y[0]
                        # mask_n[:,:,1] = y[1]
                        # mask_n[:,:,2] = y[2]
                        # mask_n[:,:,0] *= mask
                        # mask_n[:,:,1] *= mask
                        # mask_n[:,:,2] *= mask
                        # x = (0.65)*x + (0.35)*mask_n
                        # delta_h = mh - ih
                        # delta_w = mw - iw
                        # top = delta_h // 2
                        # bottom = mh - (delta_h - top)
                        # left = delta_w // 2
                        # right = mw - (delta_w - left)
                        # mask_n = mask_n[top:bottom, left:right, :]
                        # image_n = image_n + x
                        durations.append(duration)
                    image = Image.fromarray(np.uint8(image_n))
                    image = ImageTk.PhotoImage(image)
                
                    if self.panel is None:
                        self.panel = tki.Label(image=image)
                        self.panel.image = image
                        self.panel.pack(side="left", padx=10, pady=10)
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image
 
        except RuntimeError:
            print("RuntimeError")
            
    def colorpick(self):
        color = askcolor((0,0,0),self.root)
        if color[0]:
            r,g,b = color[0]
            print(r,g,b)
            r = r/255.0
            g = g/255.0
            b = b/255.0
            global active
            active = 1
            global y
            y[0], y[1], y[2] = colorsys.rgb_to_hls(r, g, b)
            print(y)
        
    def setinactive(self):
        global active
        active = 0

    def onClose(self):
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=str2bool, default=False,
        help='True if GPU is being used')
args = vars(parser.parse_args())

vs = VideoStream().start()
time.sleep(2.0)

pba = PBA(vs,args)
pba.root.mainloop()