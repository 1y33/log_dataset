import ultralytics
import main
import get_model
import inference

'''
    train directly form this file. Only run the file in the IDE
    TO DO : Neptune call back maybe ? 
'''

def train():
    PATH_TO_DATA = ""
    PATH_TO_MODEL = ""
    name_run = ""
    epochs = 100
    batch_size = 100
    dropout = 0
    lr0 =1e-3
    lrf=1e-4
    imgsz = 1024
    cos_lr = True
    optimizer = "AdamW"
    save_period = 10
    cache = True


    m = get_model.Model(PATH_TO_MODEL)
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