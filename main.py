if __name__ == "__main__":
    # import os
    # from src.training.Performance import Performance
    # Performance(os.getcwd()).run()

    from src.training.Trainer import Trainer
    trainer = Trainer()
    trainer.load_latest_checkpoint()

    trainer.train(1000)
    # trainer.tree.greedy_simulation_until_game_over(trainer.strategy_storage)
    # print(trainer.tree.game)
