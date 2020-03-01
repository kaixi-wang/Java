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

    def InitUI(self):
        self.Bind(wx.EVT_PAINT, self.OnPaint) 
        # self.Centre()
        self.beginTime = datetime.datetime.now()
        self.timer = wx.Timer(self, id=1)
        self.timer.Start(33)
        self.Bind(wx.EVT_TIMER, self.OnUpdate, self.timer)
        self.Show(True)
    
    def OnUpdate(self, e):
        print("ON UPDATE")
        currentTime = datetime.datetime.now()
        print (currentTime - self.beginTime)
        self.beginTime = currentTime
        width = 480
        height = 270
        buffer = bytearray([0]) * width * height * 3
        self.bitmap = wx.Bitmap.FromBuffer(self.width, self.height, buffer)
        dc = wx.ClientDC(self) 
        dc.Clear() 
        self.frameToDisplay += 1
    
    def OnPaint(self, e):
        print("On Paint")
        brush = wx.Brush("white")  
        dc = wx.PaintDC(self) 
        dc.SetBackground(brush)  
        dc.Clear() 
        currentTime = datetime.datetime.now()
        
        # dc.DrawBitmap(self.bitmap, 0, 0, True) 
        for i in range(0, 500):
            dc.DrawCircle(0, 0, 500 - i)
            
        newTime = datetime.datetime.now()
        print (newTime - currentTime)
    
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


    
   
if __name__ == '__main__':
    app = wx.App()
    frm = HelloFrame(None, title='Hello World 2', size=(960, 600))
    frm.Show()

    app.MainLoop()