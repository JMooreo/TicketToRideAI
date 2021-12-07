if __name__ == "__main__":
    # import os
    # from src.training.Performance import Performance
    # Performance(os.getcwd()).run()

    from src.training.Trainer import Trainer
    trainer = Trainer()
    trainer.load_latest_checkpoint()
    trainer.train(1)
    trainer.display_strategy()
