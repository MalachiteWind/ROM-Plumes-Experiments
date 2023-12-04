import mitosis
import simple_example
import os
from pathlib import Path

# folder = "simple_example_folder"

# path_isExist = os.path.exists(folder)
# if not path_isExist:
#     os.makedirs(folder)
# abs_path = Path(folder).resolve()

folder = Path(".").resolve()

params = [
    mitosis.Parameter("my_variant", "amplitude", 4),
    
]

mitosis.run(
    simple_example, 
    params=params, 
    debug=True,
    trials_folder=folder
)