import get_model
from args_configuration import LogDatasetConfig

def train():
    args = LogDatasetConfig

    m = get_model.Model(args.PATH_TO_MODEL)
    m.add_callbacks(args.project_name, args.experiment_name, args.tags)
    m.get_dataset(args.PATH_TO_DATA)
    m.train_model(args)


train()
