import os
import logzero
from logzero import logger
import time


def which_day():
    return time.strftime('%Y-%m-%d', time.localtime(time.time()))

# random state
random_state = 9527

# project directory
project_dir = os.path.dirname(__file__)

# dataset directory
data_dir = os.path.join(project_dir,'data')

# pretrain mdoels directory
pretrain_model_dir = os.path.join(project_dir,"pretrain_models")

# logger
log_path = os.path.join(project_dir, 'logs')
log_file = os.path.join(log_path, "all_{}.log".format(which_day()))
logzero.logfile(log_file, maxBytes=1e6, backupCount=3)
