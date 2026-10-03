import cv2 as cv


# Input and feature tracking
SCALE = 1
MAX_POINTS = 300
MIN_POINTS = 150
MIN_DIST = 10
EDGE_MARGIN = 25
FB_THRESH = 1.0

LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Roll and flow display
LINE_GAIN = 3.0
ROLL_LEAK = 1.0
SHOW_GRADIENT = True  # Show the tracking/gradient image window.
SHOW_ROLL = False      # Show roll information in the flow window.
LIVE_CAPTURE = False   # Webcam input with a transparent desktop overlay.

# Edge image
BLUR_SIGMA = 0.8
EDGE_HI = 40.0
EDGE_LO = 15.0
MIN_COMPONENT = 30
EDGE_GAIN = 6.0
GRAY_W = 0.6

# FOE heatmap
FOE_MIN_FLOW = 0
FOE_MIN_PTS = 12
HEAT_GRID = 64
HEAT_SIGMA = 4.0
HEAT_SMOOTH = 0.7
HEAT_ALPHA = 0.55
HEAT_DRAW = 0.85  # Also the minimum heat used to estimate the FOE.
