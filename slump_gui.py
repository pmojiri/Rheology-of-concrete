#!/usr/bin/env python

######################################################################################
# File name : slump_gui.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: The GUI file
#
######################################################################################
#wildcard = "Python source (*.py)|*.py|" \"All files (*.*)|*.*"
# import the necessary packages
import sys
import cv2
import os
import wx
import wx.lib.agw.multidirdialog as MDD
import numpy as np
sys.path.append("/home/viki/Desktop/Slump_test/packages")
from SCC_test import SCC_Test
from SLUMP_test import SLUMP_Test
#
######################################################################################

class SLUMP(wx.App):
  def __init__(self, redirect=False):
    wx.App.__init__(self, redirect, None)
    self.frame = wx.Frame(None, title='SLUMP test')
    self.panel = wx.Panel(self.frame)
    self.createWidgets()
    self.frame.Show()
    self.plate_color_min = [0, 0, 0]
    self.plate_color_max = [179, 255, 150]  
    self.plate_size = 1000
    wx.InitAllImageHandlers()
    self.currentDirectory = os.getcwd()

  def Path(self, event) :
    """
    Create and show the Open FileDialog
    """
    dlg = wx.FileDialog(None, message="Choose a file", defaultDir = self.currentDirectory, defaultFile = "", wildcard = "*.*", style = wx.OPEN)
    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        print "You chose the following file(s):"
        print  os.path.basename(path)
        self.path.SetValue(path)
    ''' the path of image/video file'''
    self.file_path = self.path.GetValue()
    print self.file_path
    dlg.Destroy()

  def PlateSize(self, event) :
    ''' the plate size'''
    self.plate_size = self.platesize.GetValue()
    print self.plate_size

  def PlateColour(self, event) :
    ''' the plate tapped colour'''
    self.plate_colour = self.platecolour.GetValue()
    if self.plate_colour == "black":
	self.plate_color_min = [0, 0, 0]
	self.plate_color_max = [179, 255, 150]  
    if self.plate_colour == "green":
	self.plate_color_min = [60 - 15, 100, 100]
	self.plate_color_max = [60 + 15, 255, 255] 
    if self.plate_colour == "blue":
	self.plate_color_min = [120 - 15, 50, 50]
	self.plate_color_max = [120 + 15, 255, 255] 
    print self.plate_colour, self.plate_color_min, self.plate_color_max

  def ConeColour(self, event) :
    ''' the cone tapped colour'''
    self.cone_colour = self.conecolour.GetValue()
    print self.cone_colour


  def SCCImage(self, event) :
    ''' call SCC image main package'''
    self.situation.SetValue(str('SCC image test are requested!'))
    scc = SCC_Test(self.file_path, self.plate_size, self.plate_color_min, self.plate_color_max)
    diameter = scc.scc_image()
    self.ew_diameter.SetValue(str(diameter))
    self.situation.SetValue(str('The results have been saved in an excel file located at excel_files/SCC_data_image.xls'))


  def SCCVideo(self, event) :
    ''' call SLUMP image main package'''
    self.situation.SetValue(str('SCC video test are requested!'))
    scc = SCC_Test(self.file_path, self.plate_size, self.plate_color_min, self.plate_color_max)
    diameter, time = scc.scc_video()
    self.ew_diameter.SetValue(str(diameter))
    self.time.SetValue(str(time))
    self.situation.SetValue(str('The results have been saved in an excel file located at excel_files/SCC_data_video.xls'))

  def SLUMPImage(self, event) :
    ''' call SLUMP image main package'''
    self.situation.SetValue(str('SLUMP image test are requested!'))
    slump = SLUMP_Test(self.file_path, self.plate_size, self.plate_color_min, self.plate_color_max)
    diameter, height = slump.slump_image()
    self.ew_diameter.SetValue(str(diameter))
    self.height.SetValue(str(height))
    self.situation.SetValue(str('The results have been saved in an excel file located at excel_files/SLUMP_data_image.xls'))


  def SLUMPVideo(self, event) :
    ''' call SLUMP image main package'''
    self.situation.SetValue(str('SLUMP video test are requested!'))
    slump = SLUMP_Test(self.file_path, self.plate_size, self.plate_color_min, self.plate_color_max)
    diameter, height, time = slump.slump_video()
    self.ew_diameter.SetValue(str(diameter))
    self.height.SetValue(str(height))
    self.time.SetValue(str(time))
    self.situation.SetValue(str('The results have been saved in an excel file located at excel_files/SLUMP_data_video.xls'))

  
  def createWidgets(self):
    img = wx.EmptyImage(800, 0)
    self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.BitmapFromImage(img))

    Line_0 = wx.StaticText(self.panel, label='Please submit all the requested information')

    Line_1 = wx.StaticText(self.panel, label='image/video path:     ')
    self.path = wx.TextCtrl(self.panel, size=(450,-1))
    self.path.SetValue(str('Please browse the image/video file'))
    Submit_path = wx.Button(self.panel, label='Browse ...')
    Submit_path.Bind(wx.EVT_BUTTON, self.Path)

    Line_2 = wx.StaticText(self.panel, label='plate size (mm):          ')
    self.platesize = wx.TextCtrl(self.panel, size=(450,-1))
    self.platesize.SetValue(str(1000))
    Submit_platesize = wx.Button(self.panel, label='Submit')
    Submit_platesize.Bind(wx.EVT_BUTTON, self.PlateSize)


    Line_3 = wx.StaticText(self.panel, label='select black or green or blue ')
    Line_3_1 = wx.StaticText(self.panel, label='plate tapped colour: ')
    self.platecolour = wx.TextCtrl(self.panel, size=(450,-1))
    self.platecolour.SetValue(str('Please insert the plate tappped colour'))
    Submit_platecolour = wx.Button(self.panel, label='Submit')
    Submit_platecolour.Bind(wx.EVT_BUTTON, self.PlateColour)


    #Line_4 = wx.StaticText(self.panel, label='cone tapped colour:  ')
    #self.conecolour = wx.TextCtrl(self.panel, size=(450,-1))
    #self.conecolour.SetValue(str('Please insert the cone tappped colour'))
    #Submit_conecolour = wx.Button(self.panel, label='Submit')
    #Submit_conecolour.Bind(wx.EVT_BUTTON, self.ConeColour)

    Line_5 = wx.StaticText(self.panel, label='Please select one of bellow test types: ')

    SCC_image = wx.Button(self.panel, label='  SCC-image test  ')
    SCC_image.Bind(wx.EVT_BUTTON, self.SCCImage)

    SCC_video = wx.Button(self.panel, label='  SCC-video test  ')
    SCC_video.Bind(wx.EVT_BUTTON, self.SCCVideo)

    SLUMP_image = wx.Button(self.panel, label='  SLUMP-image test  ')
    SLUMP_image.Bind(wx.EVT_BUTTON, self.SLUMPImage)

    SLUMP_video = wx.Button(self.panel, label='  SLUMP-video test  ')
    SLUMP_video.Bind(wx.EVT_BUTTON, self.SLUMPVideo)


    Line_7 = wx.StaticText(self.panel, label='Results:  ')
    Line_7_1 = wx.StaticText(self.panel, label=' East-West diameter (mm) ')
    self.ew_diameter = wx.TextCtrl(self.panel, size=(100,-1))
    self.ew_diameter.SetValue(str(0.0))
    Line_7_2 = wx.StaticText(self.panel, label=' Height (mm) ')
    self.height = wx.TextCtrl(self.panel, size=(100,-1))
    self.height.SetValue(str(0.0))
    Line_7_3 = wx.StaticText(self.panel, label=' Test time (ms) ')
    self.time = wx.TextCtrl(self.panel, size=(100,-1))
    self.time.SetValue(str(0.0))

    Line_8 = wx.StaticText(self.panel, label='Situation: ')
    self.situation = wx.TextCtrl(self.panel, size=(600,-1))
    self.situation.SetValue(str('I am waiting...')) 

   
    self.mainSizer = wx.BoxSizer(wx.VERTICAL)
    self.sizer0 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer1 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer3 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer3_1 = wx.BoxSizer(wx.HORIZONTAL)
    #self.sizer4 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer5 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer6 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer7 = wx.BoxSizer(wx.HORIZONTAL)
    self.sizer8 = wx.BoxSizer(wx.HORIZONTAL)


    self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)

    self.sizer0.Add(Line_0, 0, wx.ALL|wx.CENTER, 5)

    self.sizer1.Add(Line_1, 0, wx.ALL, 5)
    self.sizer1.Add(self.path, 0, wx.ALL, 5)
    self.sizer1.Add(Submit_path, 0, wx.ALL, 5)

    self.sizer2.Add(Line_2, 0, wx.ALL, 5)
    self.sizer2.Add(self.platesize, 0, wx.ALL, 5)
    self.sizer2.Add(Submit_platesize, 0, wx.ALL, 5)

    self.sizer3.Add(Line_3, 0, wx.ALL, 5)
    self.sizer3_1.Add(Line_3_1, 0, wx.ALL, 5)
    self.sizer3_1.Add(self.platecolour, 0, wx.ALL, 5)
    self.sizer3_1.Add(Submit_platecolour, 0, wx.ALL, 5)

    #self.sizer4.Add(Line_4, 0, wx.ALL, 5)
    #self.sizer4.Add(self.conecolour, 0, wx.ALL, 5)
    #self.sizer4.Add(Submit_conecolour, 0, wx.ALL, 5)

    self.sizer5.Add(Line_5, 0, wx.ALL|wx.CENTER, 5)

    self.sizer6.Add(SCC_image, 0, wx.ALL, 5)
    self.sizer6.Add(SCC_video, 0, wx.ALL, 5)
    self.sizer6.Add(SLUMP_image, 0, wx.ALL, 5)
    self.sizer6.Add(SLUMP_video, 0, wx.ALL, 5)

    self.sizer7.Add(Line_7, 0, wx.ALL, 5)
    self.sizer7.Add(Line_7_1, 0, wx.ALL, 5)
    self.sizer7.Add(self.ew_diameter, 0, wx.ALL, 5)
    self.sizer7.Add(Line_7_2, 0, wx.ALL, 5)
    self.sizer7.Add(self.height, 0, wx.ALL, 5)
    self.sizer7.Add(Line_7_3, 0, wx.ALL, 5)
    self.sizer7.Add(self.time, 0, wx.ALL, 5)
    
    self.sizer8.Add(Line_8, 0, wx.ALL, 5)
    self.sizer8.Add(self.situation, 0, wx.ALL, 5)

    self.mainSizer.Add(self.sizer0, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer1, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer2, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer3, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer3_1, 0, wx.ALL, 5)
    #self.mainSizer.Add(self.sizer4, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer5, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer6, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer7, 0, wx.ALL, 5)
    self.mainSizer.Add(self.sizer8, 0, wx.ALL, 5)

    self.panel.SetSizer(self.mainSizer)
    self.mainSizer.Fit(self.frame)
    self.panel.Layout()

if __name__ == '__main__':
  app = SLUMP()
  app.MainLoop()
    
  
