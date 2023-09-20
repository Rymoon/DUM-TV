from .main import Model,compile_training_tv_ctrl

class Model_ctrl(Model):
    """
    """
    pass

# PLOT config        
from dum_tv.utils import get_configs_path
configs_path = get_configs_path(pkg,__file__)


# PLOT compile_xxx
import pytorch_lightning as pl
from dum_tv.utils import get_root_Results
import dum_tv as pkg
root_Results = get_root_Results(pkg)
assert (root_Results).exists(),f"Results folder not exists. Create of softlink it: {root_Results}"

compile_training_tv_ctrl(*args,**kwargs):
    return compile_training_tv_ctrl(*args,**kwargs)


