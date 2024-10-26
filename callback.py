import ultralytics
import main
import inference
import train_model_file
import neptune
'''
    Logica din spate sa o inteleg mai bine:
    -> trebuie sa salvam la fiecare N epoci fiecare wieghts-urile
    -> metricurile sa le trimitem pe neptune.ai pentru a vedea rezultatele
    -> salvare metrici
    -> salvare model
    
'''


class own_callback:
    def __init__(self,project_name,run_name,save_interval):
        self.project_name = project_name
        self.run_name = run_name

        self.save_interval = save_interval

    def on_epoch_end(self, epoch,metrics):
        if epoch % self.save_interval == 0:
            model_path = f"{self.run_name}/model_epoch{epoch}.pt"
            self.save_model(model_path)
            self.neptune_run["model"].upload(model_path)

    def save_model(self,model_path):
        self.model.save(model_path)



def on_train_epoch_end(trainer):

