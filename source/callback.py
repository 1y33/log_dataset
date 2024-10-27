import neptune
from neptune.types import File
import os

# '''
#     Logica din spate sa o inteleg mai bine:
#     -> trebuie sa salvam la fiecare N epoci fiecare wieghts-urile
#     -> metricurile sa le trimitem pe neptune.ai pentru a vedea rezultatele
#     -> salvare metrici
#     -> salvare model
#
# '''
def _log_scalars(scalars, step=0):
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


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


def _save_weights(trainer, best=True, last=True):
    if (trainer.epoch + 1) % 10 == 0:
        save = f"Configuration/Model/Epoch_{trainer.epoch + 1}.pt"
        if best == True:
            run[save].upload(File(str(trainer.best)))
        else:
            run[save].upload(File(str(trainer.last)))

    if trainer.epoch+1 == trainer.epochs:
        run["Configuration/Model/best_final.pt"].upload(File(str(trainer.best)))
        run["Configuration/Model/last.pt"].upload(File(str(trainer.best)))


def on_pretrain_routine_start(trainer, project_name, experiment_name, tags):
    global run
    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    run = neptune.init_run(
        project=project_name,
        name=experiment_name,
        tags=tags,
        api_token=api_token
    )
    run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}


def on_train_epoch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")
    _save_weights(trainer)


def on_fit_epoch_end(trainer):
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers
        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_train_end(trainer):
    """Callback function called at end of training."""
    if run:
        # Log final results, CM matrix + PR plots
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # filter
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        # Log the final model
        _save_weights(trainer)
        run.stop()


def create_callbacks(project_name, experiment_name, tags):
    def on_pretrain_routine_start_with_args(trainer):
        on_pretrain_routine_start(trainer, project_name, experiment_name, tags)

    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start_with_args,  # Use the wrapper here
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
        "on_fit_epoch_end": on_fit_epoch_end
    }

    return callbacks