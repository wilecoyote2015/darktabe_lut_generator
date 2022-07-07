import PyOpenColorIO as OCIO

# Load an existing configuration from the environment.
# The resulting configuration is read-only. If $OCIO is set, it will use that.
# Otherwise it will use an internal default.
config = OCIO.GetCurrentConfig()

# What color spaces exist?
colorSpaceNames = [cs.getName() for cs in config.getColorSpaces()]
print(colorSpaceNames)

# Given a string, can we parse a color space name from it?
inputString = 'myname_linear.exr'
colorSpaceName = config.parseColorSpaceFromString(inputString)
if colorSpaceName:
    print('Found color space', colorSpaceName)
else:
    print('Could not get color space from string', inputString)

# What is the name of scene-linear in the configuration?
colorSpace = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR)
if colorSpace:
    print(colorSpace.getName())
else:
    print('The role of scene-linear is not defined in the configuration')

# For examples of how to actually perform the color transform math,
# see 'Python: Processor' docs.

# Create a new, empty, editable configuration
config = OCIO.Config()

# For additional examples of config manipulation, see
# https://github.com/imageworks/OpenColorIO-Configs/blob/master/nuke-default/make.py
