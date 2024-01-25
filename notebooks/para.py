# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import sys

# If we're not handing them in on CLI, make sure these are populated already (ie, if running interactively in Paraview...)
if len(sys.argv) > 1:
    project_name=sys.argv[1]
    im_text = sys.argv[2]
    base_dir = sys.argv[3]
    
print(f"para-processing {project_name}...")

# create a new 'XML Unstructured Grid Reader'
lcplane000flowvtu = XMLUnstructuredGridReader(registrationName=f"{project_name}-flow.vtu", FileName=[f"{base_dir}/{project_name}-flow.vtu"])
lcplane000flowvtu.PointArrayStatus = ['Density', 'Momentum', 'Energy', 'Pressure', 'Temperature', 'Mach', 'Pressure_Coefficient']

# Properties modified on lcplane000flowvtu
lcplane000flowvtu.PointArrayStatus = ['Momentum', 'Pressure']
lcplane000flowvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
lcplane000flowvtuDisplay = Show(lcplane000flowvtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
lcplane000flowvtuDisplay.Representation = 'Surface'
lcplane000flowvtuDisplay.ColorArrayName = [None, '']
lcplane000flowvtuDisplay.SelectTCoordArray = 'None'
lcplane000flowvtuDisplay.SelectNormalArray = 'None'
lcplane000flowvtuDisplay.SelectTangentArray = 'None'
lcplane000flowvtuDisplay.OSPRayScaleArray = 'Momentum'
lcplane000flowvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
lcplane000flowvtuDisplay.SelectOrientationVectors = 'Momentum'
lcplane000flowvtuDisplay.ScaleFactor = 8.8
lcplane000flowvtuDisplay.SelectScaleArray = 'Momentum'
lcplane000flowvtuDisplay.GlyphType = 'Arrow'
lcplane000flowvtuDisplay.GlyphTableIndexArray = 'Momentum'
lcplane000flowvtuDisplay.GaussianRadius = 0.44
lcplane000flowvtuDisplay.SetScaleArray = ['POINTS', 'Momentum']
lcplane000flowvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
lcplane000flowvtuDisplay.OpacityArray = ['POINTS', 'Momentum']
lcplane000flowvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
lcplane000flowvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
lcplane000flowvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
lcplane000flowvtuDisplay.ScalarOpacityUnitDistance = 1.9996234294819066
lcplane000flowvtuDisplay.OpacityArrayName = ['POINTS', 'Momentum']
lcplane000flowvtuDisplay.SelectInputVectors = ['POINTS', 'Momentum']
lcplane000flowvtuDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
lcplane000flowvtuDisplay.ScaleTransferFunction.Points = [-0.2120128870010376, 0.0, 0.5, 0.0, 1.0691198110580444, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
lcplane000flowvtuDisplay.OpacityTransferFunction.Points = [-0.2120128870010376, 0.0, 0.5, 0.0, 1.0691198110580444, 1.0, 0.5, 0.0]

# reset view to fit data
# renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
# renderView1.Update()

# set scalar coloring
ColorBy(lcplane000flowvtuDisplay, ('POINTS', 'Pressure'))

# rescale color and/or opacity maps used to include current data range
lcplane000flowvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
lcplane000flowvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')

# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction('Pressure')

# get 2D transfer function for 'Pressure'
pressureTF2D = GetTransferFunction2D('Pressure')

# create a new 'Stream Tracer'
streamTracer1 = StreamTracer(registrationName='StreamTracer1', Input=lcplane000flowvtu,
    SeedType='Line')
streamTracer1.Vectors = ['POINTS', 'Momentum']
streamTracer1.MaximumStreamlineLength = 88.0

# init the 'Line' selected for 'SeedType'
streamTracer1.SeedType.Point1 = [0, -8.0, -0.2]
streamTracer1.SeedType.Point2 = [0.0, 8.0, -0.2]

# show data in view
streamTracer1Display = Show(streamTracer1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
streamTracer1Display.Representation = 'Surface'
streamTracer1Display.ColorArrayName = ['POINTS', 'Pressure']
streamTracer1Display.LookupTable = pressureLUT
streamTracer1Display.SelectTCoordArray = 'None'
streamTracer1Display.SelectNormalArray = 'None'
streamTracer1Display.SelectTangentArray = 'None'
streamTracer1Display.OSPRayScaleArray = 'AngularVelocity'
streamTracer1Display.OSPRayScaleFunction = 'PiecewiseFunction'
streamTracer1Display.SelectOrientationVectors = 'Normals'
streamTracer1Display.ScaleFactor = 8.799880599975586
streamTracer1Display.SelectScaleArray = 'AngularVelocity'
streamTracer1Display.GlyphType = 'Arrow'
streamTracer1Display.GlyphTableIndexArray = 'AngularVelocity'
streamTracer1Display.GaussianRadius = 0.4399940299987793
streamTracer1Display.SetScaleArray = ['POINTS', 'AngularVelocity']
streamTracer1Display.ScaleTransferFunction = 'PiecewiseFunction'
streamTracer1Display.OpacityArray = ['POINTS', 'AngularVelocity']
streamTracer1Display.OpacityTransferFunction = 'PiecewiseFunction'
streamTracer1Display.DataAxesGrid = 'GridAxesRepresentation'
streamTracer1Display.PolarAxes = 'PolarAxesRepresentation'
streamTracer1Display.SelectInputVectors = ['POINTS', 'Normals']
streamTracer1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamTracer1Display.ScaleTransferFunction.Points = [-10.42719843533777, 0.0, 0.5, 0.0, 5.656404272293395, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamTracer1Display.OpacityTransferFunction.Points = [-10.42719843533777, 0.0, 0.5, 0.0, 5.656404272293395, 1.0, 0.5, 0.0]

# hide data in view
Hide(lcplane000flowvtu, renderView1)

# show color bar/color legend
streamTracer1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(lcplane000flowvtu)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=streamTracer1.SeedType)

# show data in view
lcplane000flowvtuDisplay = Show(lcplane000flowvtu, renderView1, 'UnstructuredGridRepresentation')

# show color bar/color legend
lcplane000flowvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2022, 1454)

# current camera placement for renderView1
renderView1.CameraPosition = [-5, 3, 0]
#renderView1.CameraViewUp = [0.008, 0.9688985990904215, -0.27108711464643994]
#renderView1.CameraParallelScale = 1.0 #65.1766829472013

#renderView1.AdjustAzimuth(0.0)
renderView1.AdjustElevation(-10.0)
renderView1.AdjustRoll(90.0)

text1 = Text()
text1.Text = im_text
text1Display = Show(text1, renderView1, 'TextSourceRepresentation')


# save screenshot
SaveScreenshot(f"/Users/scot/Projects/Personal/lightningcatcher/notebooks/{project_name}-para.png", renderView1, ImageResolution=[2022, 1454])
