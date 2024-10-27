import get_model


def get_hyper_parameters():
    n_epochs = [70, 80, 100]
    batch_size = [16]
    learning_rate_begining = [0.01, 0.001]
    learning_rate_final = [0.005, 0.0005]
    lr = list(zip(learning_rate_begining, learning_rate_final))
    return [n_epochs, batch_size, lr]


def hyper_params_training(path):
    SEED = 36
    DROPOUT = 0.3
    NUM_WORKERS = 8

    params = get_hyper_parameters()
    for epochs in params[0]:
        for batch_size in params[1]:
            for lrs in params[2]:
                m = get_model.Model()
                m.get_dataset(path)
                results = m.train_model(
                    run_name="searching",
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=SEED,
                    lr0=lrs[0],
                    lrf=lrs[1],
                    dropout=DROPOUT,
                    workers=NUM_WORKERS,
                    imgsz=256)
