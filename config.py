'''
file for all tunable settings
'''

HAND_LANDMARKER_MODEL    = "assets/hand_landmarker.task"

# MediaPipe detection settings
# Lowered from 0.75 → extraction uses 0.3 
# keeping inference at 0.6 reduces the train/inference distribution gap.
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE  = 0.6
MAX_NUM_HANDS            = 1

# Smoothing
LANDMARK_WINDOW_SIZE     = 5        # frames to average landmarks over
PREDICTION_WINDOW_SIZE   = 7        # frames to vote prediction over

# Classifier
MODEL_PATH = "code/static/models/sign_model.pkl"

MIN_PREDICTION_CONFIDENCE = 0.55

# Display
FONT_SCALE     = 1.4
FONT_COLOR     = (0, 255, 0)        # green
FONT_THICKNESS = 2
SHOW_LANDMARKS = True

# Dataset / extraction
DATASET_DIR      = "DATASETS/asl_alphabet_train"
DATA_CSV         = "data/hand_data.csv"
IMAGES_PER_CLASS = None           
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Training
MODELS_DIR     = "code/static/models"
FIGURES_DIR    = "code/static/figures"
PREFERRED_MODEL = "best"           

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]