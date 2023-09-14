from ren_utils.rennet import BufferDict

from pathlib import Path
def get_configs_path(pkg, env__file__):
    root_pkg = Path(pkg.__file__).parent
    configs_fname = ".".join(Path(env__file__).relative_to(root_pkg).parts)
    configs_fname  = Path(configs_fname).stem+".yaml"
    configs_path = Path(root_pkg.parent,"Scripts",configs_fname)
    return configs_path

def get_root_Results(pkg):
    root_pkg = Path(pkg.__file__).parent
    root_Results = Path(root_pkg.parent,"Results")
    assert (root_Results).exists(),f"Results folder not exists. Create of softlink it: {root_Results}"
    
    return root_Results