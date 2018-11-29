# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,LassoSelector,TextBox,CheckButtons
from matplotlib.path import Path
import numpy as np
import pickle,os
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("-f","--image_file",help="image file to mark")
parser.add_argument("-r","--ref_file",help="ref file to mark")
args = parser.parse_args()

def pad1(arr):
    a=np.ones((arr.shape[0],arr.shape[1]+1))
    a[:,:-1]=arr
    return a

def pick_colors_pts():
    start_point=np.array((242.0,193))
    stepx=(923-111.0)/3.0
    stepy=(1388-52)/5.0
    pts=[]
    for j in range(4):
        for i in range(6):
            pts.append(start_point+(stepx*(3-j),stepy*i))
    return np.array(pts) 

def pick_colors(im,pts,ws=3):
    colors=[]
    for pt in pts:
        val=im[int(pt[1])-ws:int(pt[1])+ws,int(pt[0])-ws:int(pt[0])+ws,:].reshape(-1,3).mean(axis=0)
        colors.append(val)
    return np.array(colors)

def apply_mat(img,c_mat):
    return  (img.reshape((-1,3)) @ c_mat.T).clip(0,255).reshape(img.shape).astype('uint8')

def convert_points(cross_pts1,cross_pts2,pts):
    M=np.linalg.lstsq(pad1(cross_pts1),pad1(cross_pts2),rcond=None)[0]
    return (M.T @ pad1(pts).T)[:2,:].T

class CrossMark:
    def __init__(self,img_name,imgref_name):
        self.img=cv2.imread(img_name)[:,:,::-1]
        self.imgr=cv2.imread(imgref_name)[:,:,::-1]

        fig=plt.figure(str('coross mark'))
        self.sfigs=[]
        self.sfigs+=[plt.subplot(1,2,1)]

        plt.imshow(self.imgr)
        self.sfigs+=[plt.subplot(1,2,2)]
        plt.imshow(self.img)

        axsave = plt.axes([0.8, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(self.save)

        axadd = plt.axes([0.7, 0.05, 0.1, 0.075])
        badd = Button(axadd, 'Add')
        badd.on_clicked(self.add)

        axdel = plt.axes([0.6, 0.05, 0.1, 0.075])
        bdel = Button(axdel, 'Del')
        bdel.on_clicked(self.delete)

        axrot = plt.axes([0.5, 0.05, 0.1, 0.075])
        brot = Button(axrot, 'Rot')
        brot.on_clicked(self.rotate_im)

        axplus = plt.axes([0.4, 0.05, 0.1, 0.075])
        bplus = Button(axplus, '+')
        bplus.on_clicked(self.plus)

        axmin = plt.axes([0.3, 0.05, 0.1, 0.075])
        bmin = Button(axmin, '-')
        bmin.on_clicked(self.minus)

        axchk = plt.axes([0.2, 0.90, 0.2, 0.1])
        self.chk_colors = CheckButtons(axchk,('Show_pts',),actives=[False])
        self.chk_colors.on_clicked(self.redraw)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.objects_hdls=None
        self.cross_list=[[],[]]
        self.selected_obj_ind=-1
        
        self.draw_objs()
        plt.show()

    def plus(self,event):
        self.selected_obj_ind+=1
        self.selected_obj_ind = min((self.selected_obj_ind,len(self.cross_list[0])))
        print('self.selected_obj_ind',self.selected_obj_ind)
        self.draw_objs()
        plt.draw()
    def minus(self,event):
        self.selected_obj_ind-=1
        self.selected_obj_ind = max((-1,self.selected_obj_ind))
        print('self.selected_obj_ind',self.selected_obj_ind)
        self.draw_objs()
        plt.draw()


    def rotate_im(self,event):
        self.img = cv2.transpose(cv2.flip( self.img, 1 ))
        self.cross_list[1]=[(y,self.img.shape[0]-x-1) for x,y in self.cross_list[1]]

        self.sfigs[1].cla()
        self.sfigs[1].imshow(self.img)
        self.draw_objs()  
        plt.draw()

    def redraw(self,event):
        self.draw_objs()
        plt.draw()

    def draw_objs(self,selected=-1):
        if self.objects_hdls is not None:
            for h in self.objects_hdls:
                #print('----',h)
                try:
                    h[0].remove()
                except ValueError:
                    pass
        self.objects_hdls=[]

        h=self.objects_hdls
        
        if self.chk_colors.get_status()[0]:
            pts_r = pick_colors_pts()
            pts = convert_points(np.array(self.cross_list[0]),np.array(self.cross_list[1]),pts_r)
            h.append(self.sfigs[0].plot(pts_r[:,0],pts_r[:,1],'+g'))    
            h.append(self.sfigs[1].plot(pts[:,0],pts[:,1],'+g'))    


        self.sfigs[0].set_title('{}/{}'.format(self.selected_obj_ind+1,len(self.cross_list[0])))

        for find,subp in enumerate(self.sfigs):
            #if len(self.cross_list[find]):
            #    crosses = np.array(self.cross_list[find]) 
            #    h.append(subp.plot(crosses[:,0],crosses[:,1],'+b'))
            for i,cr in enumerate(self.cross_list[find]):
                h.append(subp.plot(cr[0],cr[1],'+r' if self.selected_obj_ind==i else '+b'))

    def get_closest(self,x,y,ax_ind):
        max_ind=-1
        max_val=30 #minmal distance to accept click
        for ind,cr in enumerate(self.cross_list[ax_ind]):
            d = abs(cr[0]-x) + abs(cr[1]-y)
            if d< max_val:
                max_val=d
                max_ind=ind
        return max_ind

    def delete(self,event):
        if self.selected_obj_ind != -1:
            for i in [0,1]:
                self.cross_list[i].pop(self.selected_obj_ind)
            self.selected_obj_ind=-1
            self.draw_objs()
            plt.draw()

    def save(self,event):
        plt.figure()
        plt.subplot(1,3,1)
        plt.title('ref')
        plt.imshow(self.imgr)

        pts_r = pick_colors_pts()
        plt.plot(pts_r[:,0],pts_r[:,1],'+')

        plt.subplot(1,3,2)
        plt.title('im corrected')
        pts = convert_points(np.array(self.cross_list[0]),np.array(self.cross_list[1]),pts_r)

        plt.plot(pts[:,0],pts[:,1],'+')
        col_r=pick_colors(self.imgr,pts_r)
        col_im = pick_colors(self.img,pts)
        CM=np.transpose(np.linalg.lstsq(col_im, col_r, rcond=None)[0])

        with open('color_mat','wb') as fd:
            pickle.dump(CM,fd)
        im1_corr =apply_mat(self.img,CM)
        plt.imshow(im1_corr)

        plt.subplot(1,3,3)
        plt.title('im1')
        plt.plot(pts[:,0],pts[:,1],'+')
        plt.imshow(self.img)
        plt.show()


    def add(self,event):
        #import pdb;pdb.set_trace()
        for i in [0,1]:
            self.cross_list[i].append((-1,-1))
        self.draw_objs()
        plt.draw()

    def onclick(self,event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #          ('double' if event.dblclick else 'single', event.button,
        #                     event.x, event.y, event.xdata, event.ydata))
        ind=self.sfigs.index(event.inaxes) if event.inaxes in self.sfigs else -1
        print('inaxes',ind)
        x,y=event.xdata,event.ydata
        if ind>-1 and self.selected_obj_ind > -1:
            self.cross_list[ind][self.selected_obj_ind]=(x,y)
#        if ind>=0:
        self.draw_objs()
        plt.draw()
            
        #self.sfigs[ind].plot(x,y,'or')
        #if ind >= 0:
        #    self.last_ind_ax=ind
        #    cind=self.get_closest(x,y,ind)
        #    self.selected_obj_ind=cind
        #    if cind>-1:
        #        self.draw_objs(cind)
        #        self.text_box.set_val(self.object_list[cind]['desc'])
        #        plt.draw()

        #plt.draw()


if __name__=="__main__":
    cr=CrossMark(args.image_file,args.ref_file)

    plt.show()

