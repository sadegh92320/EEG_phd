from typing import Any
import torch
from downstream.CNN_module import CNNmodule, EEGNet
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from functools import wraps
import timeit
from copy import deepcopy
import torch.nn.functional as F
import os
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, Recall, Precision, F1Score, ConfusionMatrix, AUROC
from torchmetrics.classification import CohenKappa
import sys
from collections import Counter
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from process_data.preprocessing import Preprocessing
from dataset import EEGdataset
import numpy as np
import optuna
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import wandb

wandb.init(
    project="downstream_eeg", 
    name = "population_split",
    note = "Training the model on the whole data with the best parameters from the kfold training and evaluating on the whole test data",
    config={
        "architecture": "EEGMAE",
        "epoch_pretrain": 5,

    }
)

MODEL_REGISTRY = {
    "CNNmodule": CNNmodule,
    
}

OPTIMIZER_REGISTRY = {
    "adam": Adam,
    "SGD": SGD
}


class EarlyStopper:
    """Early stopper class used to stop the training if no improvement is made"""
    def __init__(self, patience):
        self.patience = patience
        self.count = 0
        self.best = 0
    def should_stop(self, metric):
        #Assuming the metric is the val accuracy
        #If the accuracy decreases then count restart
        if self.best < metric:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
        if self.count > self.patience:
            return True
        else:
            return False    
    


def time(func):
        """wrapper method to the time to be recorded in all training"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            s = timeit.default_timer()
            to_return = func(*args, **kwargs)
            e = timeit.default_timer()
            print(f"time taken by {func.__name__}: {e-s}")
            return to_return
        
        return wrapper  


class TrainerDownstream:
    def __init__(self, model_name, model, optimizer, loss_fn, batch_size, config, early_stopper = EarlyStopper, train_data = None, val_data = None, test_data = None, evaluation_scheme = "population"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = model
        self.optimizer_name = optimizer
        self.loss_fn = loss_fn
        self.early_stopper = early_stopper
        self.batch_size = batch_size
        self.config = config
        self.evaluation_scheme = evaluation_scheme

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        

    def build_model(self):
        "Build a new model instance"
        model_cfg = self.config["module"][self.model_name]
        model_name = model_cfg["name"]
        model_params = model_cfg.get("params", {})

        ModelClass = MODEL_REGISTRY[model_name]
        return ModelClass(**model_params)
    
    def build_optimizer(self, model, optimizer_params):
        """Build a new optimizer instance"""
        optimizer_cfg = self.config["optimizer"][self.optimizer_name]
        optimizer_name = optimizer_cfg["name"]
        optimizer = OPTIMIZER_REGISTRY[optimizer_name]
        return optimizer(model.parameters(),**optimizer_params)

    def save_model(self, model, date):
        """Save the model to the path of interest"""
        dir = os.path.join(self.config["model_path"], self.model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(model.state_dict(), os.path.join(dir, f"_state_{date}.pth"))
        return self

    def get_params(self, model):
        """Print the model's parameters"""
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        return self
    
    def load_model(self, ):
        """Load state dict of the model"""
        #model.load_state_dict(torch.load(PATH, weights_only=True))
        pass
        
    def train_one_epoch(self, optimizer, loss_fn, model, dataloader):
        '''
        One epoch iteration of training.
        '''
        model.train()
        loss_total = 0
        for x,y in tqdm(dataloader):
            x = x.float()   

            x, y = x.to(self.device), y.long().to(self.device)
            pred = model(x)
            loss = loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total = loss.item() + loss_total
        return loss_total/len(dataloader)
    

    def get_metrics(self):
        """Set the evaluation metrics"""
        metrics = {"accuracy": Accuracy(task="multiclass", num_classes=self.config["num_classes"]),
        "recall": Recall(task = "multiclass", num_classes = self.config["num_classes"], average="macro"),
        "precision": Precision(task = "multiclass", num_classes = self.config["num_classes"], average="macro"),
        "f1_score": F1Score(task="multiclass", num_classes=self.config["num_classes"], average="macro"),
        "confusion": ConfusionMatrix(task="multiclass", num_classes=self.config["num_classes"]),
        "roc_auc": AUROC(task="multiclass",num_classes=self.config["num_classes"],average="macro"),
        "kappa": CohenKappa(task="multiclass", num_classes=self.config["num_classes"])}

        return metrics

    def update_metrics(self, metrics, pred, probs, y):
        """Update the torchmetric metrics"""
        for k,v in metrics.items():
            if k != "roc_auc":
                v.update(pred, y)
            else:
                v.update(probs,y)
        return self

    def compute_metrics(self, metrics):
        """Compute all metrics of evaluation"""
        return {k:v.compute() for k,v in metrics.items()}

    def print_metric(self, metrics):
        """Print all evaluation metrics"""
        for k, v in metrics.items():
            if k == "confusion":
                print(f"{k}:{v}")
            elif hasattr(v, "item"):
                print(f"{k}:{v.item()}")
            else:
                print(f"{k}:{v}")
    
    def predict(self, model, dataloader):
        """
        Evaluate the model.
        """
        model.eval()
        #Get all the evaluation metrics
        metrics = self.get_metrics()
        for m in metrics.values():
            m.to(self.device)
            m.reset()
        with torch.no_grad():
            for x,y in tqdm(dataloader):
                x = x.float()   
                x, y = x.to(self.device), y.long().to(self.device)
                pred = model(x)
                probs = F.softmax(pred, dim = 1)
                pred_labels = pred.argmax(dim=1)

                #Update the metrics using the evaluation result
                self.update_metrics(metrics, pred, pred_labels, y)
                
        #Compute print all metrics
        metrics_comp = self.compute_metrics(metrics)
        self.print_metric(metrics_comp)
      
        return {k:v.item() for k,v in metrics_comp.items() if k!="confusion"}


    @time
    def tune_params_cv(self, folds, trial, save = False, train_dataset = None):
        """
        Cross validation of the model with kfold.
        """

        kf = KFold(n_splits=folds, shuffle=True, random_state=92)

        #Define the Optuna tuning parameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 128)
        opt_name = trial.suggest_categorical("optimizer", ["adam"])

        fold_scores = []
        fold_metrics = []
      
        num_epochs = 50
   
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            #Define early stopper for the fold
            early_stopper = self.early_stopper(patience=20)

            print("{:^100}".format(f"---Fold_{fold}---"))
            train_subs = Subset(train_dataset, train_idx)
            val_subs = Subset(train_dataset, val_idx)

            # Create DataLoaders for this specific fold
            train_loader = DataLoader(train_subs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subs, batch_size=batch_size, shuffle=False)

            #Define the best model 
            best = -float("inf")
            model = self.model().to(self.device)
            best_model = None

            #Build the optimizer and define loss function
            self.optimizer_name = opt_name
            optimizer = self.build_optimizer(model, {"lr": learning_rate})
            loss_fn = self.loss_fn
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.3, min_lr=1e-5)
            
            #Loop through epochs
            for epoch in range(num_epochs):
                #Train one epoch
                loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
                if epoch%10 == 0:
                    print(f"Epoch: {epoch} - Loss: {loss}")
                
                #Get the metrics for validation set
                metrics = self.predict(model=model, dataloader=val_loader)
                
                #Retrieve the metric of interest (mostly accuracy)
                perf = metrics[self.config["metric"]]
                scheduler.step(perf)
                wandb.log({"acc": perf, "epoch": epoch, "fold": fold})
                
                #Stop training if the val accuracy has not improved for a while
                if early_stopper.should_stop(perf):
                    break

                #If the accuaray has improved save the model
                if perf > best:
                    best_model = deepcopy(model.state_dict())
                    best  = perf
                    all_met = metrics
                    
            #Report results to Optuna and prune if necessary
            trial.report(best, step=fold)
            if trial.should_prune():
                    raise optuna.TrialPruned()
            
            fold_scores.append(best)
            fold_metrics.append(all_met)

            if save == True:
                self.save_model(model, fold)

        mean_score = float(np.mean(fold_scores))
        print(f"mean fold score is {mean_score}")
        wandb.log({"mean_fold_score": mean_score})
        accuracy = [met["accuracy"] for met in fold_metrics]
        recall = [met["recall"] for met in fold_metrics]
        precision = [met["precision"] for met in fold_metrics]
        f1 = [met["f1_score"] for met in fold_metrics]
        roc_auc = [met["roc_auc"] for met in fold_metrics]
        kappa = [met["kappa"] for met in fold_metrics]
        wandb.log({"cv_accuracy": accuracy, "cv_recall": recall, "cv_precision": precision, 
                   "cv_f1": f1, "cv_roc_auc": roc_auc, "cv_kappa": kappa})
        return mean_score
    
    @time
    def tune_params(self, trial, train_dataset = None, val_dataset = None):
       
        #Define the Optuna tuning parameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 128)
        opt_name = trial.suggest_categorical("optimizer", ["adam"])

        fold_scores = []
      
        num_epochs = 20
   

        early_stopper = self.early_stopper(patience=10)        

        # Create DataLoaders for this specific fold
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #Define the best model 
        best = -float("inf")
        model = self.model().to(self.device)

        #Build the optimizer and define loss function
        self.optimizer_name = opt_name
        optimizer = self.build_optimizer(model, {"lr": learning_rate})
        loss_fn = self.loss_fn
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.3, min_lr=1e-5)
            
            #Loop through epochs
        for epoch in range(num_epochs):
            #Train one epoch
            loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
            if epoch%10 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")
                
            #Get the metrics for validation set
            metrics = self.predict(model=model, dataloader=val_loader)
                
            #Retrieve the metric of interest (mostly accuracy)
            perf = metrics[self.config["metric"]]
            wandb.log({"acc": perf, "epoch": epoch})
            scheduler.step(perf)
                
            #Stop training if the val accuracy has not improved for a while
            if early_stopper.should_stop(perf):
                break

            if perf > best:
                best  = perf
            #Report results to Optuna and prune if necessary
            trial.report(best, step=epoch)
            if trial.should_prune():
                    raise optuna.TrialPruned()
        return best
    @time
    def evaluate_model(self, learning_rate, opt_name,batch_size,num_epochs,train_dataset = None, val_dataset = None, test_dataset = None):

        early_stopper = self.early_stopper(patience=20)        

        # Create DataLoaders for this specific fold
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        #Define the best model 
        best = -float("inf")
        model = self.model().to(self.device)
        best_model = model.state_dict()

        #Build the optimizer and define loss function
        self.optimizer_name = opt_name
        optimizer = self.build_optimizer(model, {"lr": learning_rate})
        loss_fn = self.loss_fn
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.3, min_lr=1e-5)
            
            #Loop through epochs
        for epoch in range(num_epochs):
            #Train one epoch
            loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
            if epoch%10 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")
                
            #Get the metrics for validation set
            metrics = self.predict(model=model, dataloader=val_loader)
                
            #Retrieve the metric of interest (mostly accuracy)
            perf = metrics[self.config["metric"]]
            scheduler.step(perf)
            if perf > best:
                best  = perf
                best_model = deepcopy(model.state_dict())
            wandb.log({"acc_eval": perf, "epoch": epoch})

                
            #Stop training if the val accuracy has not improved for a while
            if early_stopper.should_stop(perf):
                break
            #Report results to Optuna and prune if necessary
        model.load_state_dict(best_model)
        test_metrics = self.predict(model=model, dataloader=test_loader)
        return test_metrics
    

    def run_population(self, save = False):
        wandb.log({"evaluation_scheme": "population"})
        def objective(trial):
            train_dataset = ConcatDataset([self.train_data, self.val_data, self.test_data])
            return self.tune_params_cv(folds=5, trial=trial, save=save, train_dataset=train_dataset)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
       
        
        
    def run_per_subject(self, participant_number, train_data_sub, val_data_sub, test_data_pop, test_data_sub, save = False):
        wandb.log({"evaluation_scheme": "per_subject"})
        def objective(trial):
            train_dataset = ConcatDataset([train_data_sub, val_data_sub])
            return self.tune_params_cv(folds=5, trial=trial, save=save, train_dataset=train_dataset)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        lr = study.best_trial.params["learning_rate"]
        batch_size = study.best_trial.params["batch_size"]
        opt_name = study.best_trial.params["optimizer"]
        self.optimizer_name = opt_name
        num_epochs = 50

        metrics_pop = self.evaluate_model(learning_rate=lr, opt_name=opt_name, batch_size=batch_size, num_epochs=num_epochs, train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_pop)
        wandb.log({"participant_number": participant_number})
        wandb.log({"test_acc_pop": metrics_pop["accuracy"], "test_recall_pop": metrics_pop["recall"], 
                   "test_precision_pop": metrics_pop["precision"], "test_f1_pop": metrics_pop["f1_score"], 
                   "test_roc_auc_pop": metrics_pop["roc_auc"], "test_kappa_pop": metrics_pop["kappa"]})
        
        metrics_sub = self.evaluate_model(learning_rate=lr, opt_name=opt_name, batch_size=batch_size, num_epochs=num_epochs, train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_sub)
       
        wandb.log({"test_acc_sub": metrics_sub["accuracy"], "test_recall_sub": metrics_sub["recall"], 
                   "test_precision_sub": metrics_sub["precision"], "test_f1_sub": metrics_sub["f1_score"], 
                   "test_roc_auc_sub": metrics_sub["roc_auc"], "test_kappa_sub": metrics_sub["kappa"]})
        
    def run_LOSO(self, participant_number, train_data_pop, val_data_pop, test_data_sub, save = False):
        wandb.log({"evaluation_scheme": "LOSO"})
        def objective(trial):
            return self.tune_params(trial, train_dataset=train_data_pop, val_dataset=val_data_pop)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        lr = study.best_trial.params["learning_rate"]
        batch_size = study.best_trial.params["batch_size"]
        opt_name = study.best_trial.params["optimizer"]
        self.optimizer_name = opt_name
        num_epochs = 50

        metrics_sub = self.evaluate_model(learning_rate=lr, opt_name=opt_name, batch_size=batch_size, num_epochs=num_epochs, train_dataset=train_data_pop, val_dataset=val_data_pop, test_dataset=test_data_sub)
       
        wandb.log({"participant_number": participant_number})
        wandb.log({"test_acc_sub": metrics_sub["accuracy"], "test_recall_sub": metrics_sub["recall"], 
                   "test_precision_sub": metrics_sub["precision"], "test_f1_sub": metrics_sub["f1_score"], 
                   "test_roc_auc_sub": metrics_sub["roc_auc"], "test_kappa_sub": metrics_sub["kappa"]})


        

        
    @time
    def training_skf(self, dataset):
        """
        cross validation of the model with stratified kfold.
        """
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        y = dataset.label.argmax(axis=1)
    
        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.data, y)):
            model = self.build_model()
            optimizer = self.build_optimizer(model)
            loss_fn = self.loss_fn
            train_loader = DataLoader(dataset=dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            val_loader = DataLoader(dataset=dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
            
            for epoch in range(10):
                loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
                metrics = self.predict(model=model, dataloader=val_loader)
                perf = metrics[self.config["metric"]]
                if perf > best:
                    best_model = deepcopy(model.state_dict())
                    best  = perf

            if best_model is not None:
                model.load_state_dict(best_model)
            self.save_model(model, fold)

    def append_result(self, model, model_name, opt_name, study, accuracy, precision, recall, F1):
        """Write the results in a txt file"""
        date_current = date.today().strftime('%Y-%m-%d')
        time = str(datetime.now())
       
        #Prepare path and append the current model architecture
        dir = os.path.join(self.config["result_output"], date_current)
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = f"{dir}/experiment_{model_name}_{time}.txt"
        path2 = f"{dir}/architecture_{model_name}_{time}.txt"
        self.append_model_architecture(model, path2)
        
        with open(path, "a") as file:
            file_size = os.stat(path).st_size
            
            if file_size == 0:
                headers = ["Time"] \
                        + ["Model name"]\
                        + ["Optimizer name"] \
                        + list(study.keys()) \
                        + ["Accuracy"] + ["Precision"] + ["recall"] + ["F1"] + "\n"

                header_line = " ".join(f"{h:<10}" for h in headers)
                file.write(header_line)
           
            row_items = [
                    time,
                    model_name,
                    opt_name,
                    *study.values(),
                    accuracy,
                    precision,
                    recall,
                    F1,
                    "\n"
                        ]

            row_line = " ".join(f"{str(h):<10}" for h in row_items)

            file.write(row_line)

    def append_model_architecture(self, model, path):
        with open(path, "w") as f:
            f.write(repr(model))
            



    
if __name__ == "__main__":
    pass
    

    