import sys
sys.path.append("../")

from utils.clusting_utils import CoarseLeadingForest as CLF
from data.DT_Animal10N import Animal10N
from utils.logger_utils import custom_logger
from functools import reduce

clf = CLF.load("")
paths, _ = clf.generate_path()

for path in paths:
    print(path)
