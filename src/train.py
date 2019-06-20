from pathlib import Path
import torch
from SegmentationAgent import SegmentationAgent
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from tqdm import tqdm

NUM_DATA = 1000
TRAIN_NUM = 700
VAL_NUM = 200
NUM_CLASSES = 13
BATCH_SIZE = 8
IMG_SIZE = 224
DATA_PATH = Path('../data')
SHUFFLE = True
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = SegmentationAgent(NUM_DATA, TRAIN_NUM, VAL_NUM, NUM_CLASSES,
                          BATCH_SIZE, IMG_SIZE, DATA_PATH, SHUFFLE, LR, DEVICE)

trainer = create_supervised_trainer(agent.model, agent.optimizer,
                                    agent.criterion)
evaluator = create_supervised_evaluator(agent.model, metrics={
    'accuracy': Accuracy(), 'loss': Loss(agent.criterion)
})

desc = "ITERATION - loss: {:.2f}"
pbar = tqdm(initial=0, leave=False, total=len(agent.train_loader),
            desc=desc.format(0))
log_interval = 10


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(agent.train_loader) + 1

    if iter % log_interval == 0:
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    pbar.refresh()
    evaluator.run(agent.train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    tqdm.write(
        "Training - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, avg_accuracy, avg_loss)
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(agent.validation_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    tqdm.write(
        "Validation - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            engine.state.epoch, avg_accuracy, avg_loss))
    pbar.n = pbar.last_print_n = 0


trainer.run(agent.train_loader, max_epochs=20)
pbar.close()
