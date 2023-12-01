import mitosis
import simple_example
import os
from pathlib import Path

folder = "simple_example"

path_isExist = os.path.exists(folder)
if not path_isExist:
    os.makedirs(folder)
abs_path = Path(folder).resolve()

params = [
    mitosis.Parameter("my_variant", "amplitude", 4),
    
]

mitosis.run(
    simple_example, 
    params=params, 
    debug=True,
    trials_folder=abs_path
)