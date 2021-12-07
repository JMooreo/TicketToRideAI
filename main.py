import pstats
import cProfile

from src.game.enums.TrainColor import TrainColor
from src.training.Trainer import Trainer


# def main():
    # trainer.training_step()


if __name__ == "__main__":
    print(list(TrainColor))
    # trainer = Trainer()
    # trainer.load_latest_checkpoint()
    # trainer.tree.simulate_for_n_turns(2)
    #
    # cProfile.run('main()', 'output.dat')
    #
    # with open("./output_time.txt", "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("time").print_stats()
    #
    # with open("./output_calls.txt", "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("calls").print_stats()
