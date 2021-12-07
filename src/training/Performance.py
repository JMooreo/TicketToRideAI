import cProfile
import pstats

from src.training.Trainer import Trainer


def code_to_run():
    trainer = Trainer()
    trainer.load_latest_checkpoint()
    trainer.tree.simulate_for_n_turns(2)
    trainer.training_step()


class Performance:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def run(self):
        cProfile.runctx('code_to_run()', globals(), locals(), "output.dat")

        with open(f"{self.output_directory}/output_time.txt", "w") as f:
            p = pstats.Stats("output.dat", stream=f)
            p.sort_stats("time").print_stats()

        with open(f"{self.output_directory}/output_calls.txt", "w") as f:
            p = pstats.Stats("output.dat", stream=f)
            p.sort_stats("calls").print_stats()
