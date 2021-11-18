import torch

from argparse import ArgumentParser

from higgs.data import HiggsDataset
from higgs.logging import get_logger
from higgs.models import MLP
from higgs.train import train
from higgs.train import test

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from torch import nn
from torch.utils.data import DataLoader


logger = get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_higgs(config, training_loader, valid_loader, n_epochs=50):

    model = MLP()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for t in range(n_epochs):
        train(training_loader, model, loss_fn, optimizer)
        test_loss, correct = test(valid_loader, model, loss_fn)
        print(
            f"Epoch {t+1}/{n_epochs} | Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")

        tune.report(mean_accuracy=correct)

    print("Done!")


def tune_higgs(training, valid):

    search_space = {
        "lr": tune.loguniform(1e-6, 1e-1)
    }

    analysis = tune.run(tune.with_parameters(train_higgs, training_loader=training, valid_loader=valid, n_epochs=10),
                        num_samples=5,
                        scheduler=ASHAScheduler(
                            metric="mean_accuracy", mode="max"),
                        config=search_space,
                        resources_per_trial={'gpu': 1})

    # Plot by epoch
    dfs = analysis.trial_dataframes

    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)


def main(command, data_path):
    logger.info("Starting training process...")
    training_ds, valid_ds, testing_ds = HiggsDataset(data_path)
    train_lg_ds, train_sm_ds = torch.utils.data.random_split(
        training_ds, lengths=[9900000, 100000])

    train_lg_loader = DataLoader(train_lg_ds, batch_size=128*100)
    train_sm_loader = DataLoader(train_sm_ds, batch_size=128*10)
    valid_loader = DataLoader(valid_ds, batch_size=128)
    testing_loader = DataLoader(testing_ds, batch_size=128)

    del training_ds, valid_ds, testing_ds, train_lg_loader

    tune_higgs(train_sm_loader, valid_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("--command", type=str,
                        help="Path to experiment config toml file")
    parser.add_argument("--data_path", type=str,
                        help="Path to experiment config toml file")
    args = parser.parse_args()
    main(args.command, args.data_path)
