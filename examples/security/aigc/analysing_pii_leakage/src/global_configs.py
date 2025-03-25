from dataclasses import dataclass

from src.utils.output import print_warning

try:
    from src.local_configs import LocalConfigs

    system_configs = LocalConfigs()  # Make sure to implement this class.

    """ Example content of 'src.local_configs.py':
    
    from dataclasses import dataclass

    @dataclass
    class LocalConfigs:
        CACHE_DIR = "./.cache"
        IMAGENET_ROOT = "/my/path/to/imagenet"
        
    """

except Exception as e:
    print_warning(f"No local system configs found. Look into 'src.global_configs.py' to solve this problem."
                  f"Create a 'local_configs.py' file and paste it into the src folder containing a dataclass "
                  f"'LocalConfigs' with your local system configs.")

    @dataclass
    class GlobalConfigs:
        CACHE_DIR = "./.cache"  # Path to the local cache. This is where we persist hidden files.
        IMAGENET_ROOT = ""      # Path to the ImageNet dataset. it should contain two folders: 'train' and 'val'


    system_configs = GlobalConfigs()
