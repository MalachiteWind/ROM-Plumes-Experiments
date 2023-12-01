import mitosis
import simple_example
from pathlib import Path

folder = Path("simple_example").resolve()
params = [
    mitosis.Parameter("my_variant", "amplitude", 4),
    
]

mitosis.run(
    simple_example, 
    params=params, 
    debug=True,
    trials_folder=folder
)