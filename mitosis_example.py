import mitosis
import simple_example
from pathlib import Path

folder = Path(".").resolve()
params = [mitosis.Parameter("my_variant", "amplitude", 4)]

mitosis.run(simple_example, params=params, trials_folder=folder)

# print(params[0].id_name)