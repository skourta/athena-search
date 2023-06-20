from athena.tiramisu.tiramisu_actions.fusion import Fusion
from athena.tiramisu import TiramisuProgram, tiramisu_actions
from athena_search.exploration.random_exploration import RandomExploration

# from athena_search.utils.config import BaseConfig
from athena.utils.config import BaseConfig
import tests.utils as test_utils


if __name__ == "__main__":
    BaseConfig.init()

    tiramisu_program = test_utils.multiple_roots_sample()

    RandomExploration(tiramisu_program).run()
