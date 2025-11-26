import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# File names
RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"
META_FILE = "movie_metadata.csv"
GENOME_TAGS_FILE = "genome-tags.csv"
GENOME_SCORES_FILE = "genome-scores.csv"

# Modelling constants
LIKE_THRESHOLD = 4.0          # rating >= 4 => like
RANDOM_STATE = 42
N_TEXT_COMPONENTS = 50        # SVD components for overview text

NEW_RELEASE_YEAR = 2010
CURRENT_YEAR_FOR_RECENCY = 2016  # approx for MovieLens-like data
