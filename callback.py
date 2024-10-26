import neptune
from neptune.types import File

# '''
#     Logica din spate sa o inteleg mai bine:
#     -> trebuie sa salvam la fiecare N epoci fiecare wieghts-urile
#     -> metricurile sa le trimitem pe neptune.ai pentru a vedea rezultatele
#     -> salvare metrici
#     -> salvare model
#
# '''

def on_pretrain_routine_start(trainer, project_name, experiment_name, tags,api_key):
    global run
    run = neptune.init_run(
        api_token = api_key,
        project=project_name,
        name=experiment_name,
        tags=tags,
    )
    run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}

def _log_scalars(scalars, step=0):
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)

def on_train_epoch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)

def _log_images(imgs_dict, group=""):
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))

def _log_plot(title, plot_path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)

def on_train_end(trainer):
    if run:
        run.stop()

def create_callbacks(model, project_name, experiment_name, tags,api_key):
    def on_pretrain_routine_start_with_args(trainer):
        on_pretrain_routine_start(trainer, project_name, experiment_name, tags,api_key)

    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
    }

    model.add_callbacks(callbacks)
    return model

