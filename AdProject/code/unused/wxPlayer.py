#!/bin/python
"""
Hello World, but with more meat.
"""

import wx
import sys
from MyImage import MyImage 
import time
import datetime
import threading 
from multiprocessing import Pool, TimeoutError, Process, Array, Manager
import os

class HelloFrame(wx.Frame):
    """
    A Frame that says Hello World
    """
    width = 480
    height = 270
    bitmap = None
    frameToDisplay = 0
    frameToRead = 0

    def __init__(self, *args, **kw):
        manager = Manager()
        self.images = manager.dict(lock=False)

        self.pool = {}
        # ensure the parent's __init__ is called
        super(HelloFrame, self).__init__(*args, **kw)
        self.InitUI()        

        self.isPlaying = True

        self.Bind(wx.EVT_PAINT, self.OnPaint) 
        # self.Centre()
        self.beginTime = datetime.datetime.now()
        self.timer = wx.Timer(self, id=1)
        self.timer.Start(33)
        self.Bind(wx.EVT_TIMER, self.OnUpdate, self.timer)

        self.Bind(wx.EVT_IDLE, self.ReadImage)

        Size  = self.ClientSize
        self._Buffer = wx.EmptyBitmap(*Size)

    def InitUI(self):
        panel = wx.Panel(self) 
        vbox = wx.BoxSizer(orient=wx.VERTICAL) 

        vbox.AddSpacer(270)
        
        self.hPanel = wx.Panel(panel)
        hbox = wx.BoxSizer(wx.HORIZONTAL) 
        self.btn1 = wx.Button(self.hPanel,-1,"Start") 
        self.btn1.SetBackgroundColour(wx.Colour(0, 0, 0))
        hbox.Add(self.btn1,0,wx.LEFT) 

        self.btn2 = wx.Button(self.hPanel,-1,"Pause") 
        self.btn2.SetBackgroundColour(wx.Colour(0, 0, 0))
        hbox.Add(self.btn2,0,wx.LEFT) 
        self.hPanel.SetSizer(hbox) 

        self.btn3 = wx.Button(self.hPanel,-1,"Process") 
        self.btn3.SetBackgroundColour(wx.Colour(0, 0, 0))
        hbox.Add(self.btn3,0,wx.LEFT) 
        self.hPanel.SetSizer(hbox) 
        vbox.Add(self.hPanel, 0)

        panel.SetSizer(vbox) 

        self.btn1.Bind(wx.EVT_BUTTON,self.OnButtonStart) 
        self.btn2.Bind(wx.EVT_BUTTON,self.OnButtonPause) 
        self.btn3.Bind(wx.EVT_BUTTON,self.OnButtonProcess) 

        self.Show(True)
    
    def OnButtonStart(self, e):
        self.isPlaying = True
    
    def OnButtonPause(self, e):
        self.isPlaying = False
    
    def OnButtonProcess(self, e):
        self.images[self.frameToDisplay-1].TemplateMatch()

    def OnUpdate(self, e):
        # print("ON UPDATE")
        currentTime = datetime.datetime.now()
        # print (currentTime - self.beginTime)
        self.beginTime = currentTime

        if (self.isPlaying and self.images.__contains__(self.frameToDisplay)):
            # pass
            # print("Contains " + str(self.frameToDisplay))
            buffer = self.images[self.frameToDisplay].GetBuffer()
            self.bitmap = wx.Bitmap.FromBuffer(self.width, self.height, buffer)
            dc = wx.ClientDC(self) 
            dc.Clear() 
            self.frameToDisplay += 10
    
    def OnPaint(self, e):
        brush = wx.Brush("black")  
        # self.dc = wx.PaintDC(self) 
        self.dc = wx.BufferedPaintDC(self, self._Buffer)
        self.dc.Clear() 
        self.dc.SetBackground(brush)  
        if (self.bitmap != None):
            self.dc.DrawBitmap(self.bitmap, 0, 0, True) 
    
    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


    def OnHello(self, event):
        """Say hello to the user."""
        wx.MessageBox("Hello again from wxPython")


    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK|wx.ICON_INFORMATION)

    def readVideo(self, file_path):
        self.file = open(file_path, 'rb')
        file_size = os.path.getsize(file_path)
        self.maxFrame = file_size / (480 * 270 * 3)

        print("Frame Length: " + str(self.maxFrame))
    
    def ReadImage(self, e):
        if (self.frameToRead < self.maxFrame and self.frameToRead < self.frameToDisplay + 50):
            mBytes = self.file.read(480 * 270 * 3)
            p = Process(target=self.ReadOneImage, args=(mBytes, self.frameToRead, self.images))
            # self.ReadOneImage(mBytes, self.frameToDisplay, self.images)
            p.start()
            self.frameToRead += 1

    def ReadOneImage(self, byts, frame, images):
        img = MyImage()
        img.ReadImage(byts)
        images[frame] = img
        # print(img.buffer[:10])
        # print(frame)

    def SetCurrentFrame(self, frame):
        self.frameToDisplay = frame

if __name__ == '__main__':
    app = wx.App()
    frm = HelloFrame(None, title='Hello World 2', size=(480, 400))
    frm.readVideo(sys.argv[1])
    frm.SetCurrentFrame(0)
    frm.Show()

    app.MainLoop()