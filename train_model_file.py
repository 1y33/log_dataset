import get_model
import os
'''
    train directly form this file. Only run the file in the IDE
    TO DO : Neptune call back maybe ? 
'''

def train():

    PATH_TO_DATA = "data/data.yaml"
    PATH_TO_MODEL =None
    name_run = ""
    epochs = 10
    batch_size = 10
    dropout = 0
    lr0 =1e-3
    lrf=1e-4
    imgsz = 640
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = False

    project_name = "1y33/test"
    experiment_name = 'experimenting_csv'
    tags = ['Ultralytics', 'Training']


    m = get_model.Model(PATH_TO_MODEL)
    m.add_callbacks(project_name,experiment_name,tags)
    m.get_dataset(PATH_TO_DATA)

    m.train_model(
        run_name=name_run,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        lr0=lr0,
        lrf=lrf,
        imgsz=imgsz,
        cos_lr=cos_lr,
        optimizer=optimizer,
        save_period = save_period,
        cache = cache
    )

train()