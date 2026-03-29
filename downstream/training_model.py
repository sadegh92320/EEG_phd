from typing import Any
import inspect
import torch
from downstream.models.deep_learning_model.eeg_net import EEGNet as EEGNetBaseline
from downstream.models.deep_learning_model.conformer import Conformer
from downstream.models.deep_learning_model.ctnet import EEGTransformer
from downstream.models.deep_learning_model.deepconvnet import DeepConvNet
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from functools import wraps
import timeit
from copy import deepcopy
import torch.nn.functional as F
import os
from torch.optim import SGD, Adam, AdamW
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
from optuna.samplers import TPESampler
import random



MODEL_REGISTRY = {
    "EEGNetBaseline": EEGNetBaseline,
    "Conformer": Conformer,
    "EEGTransformer": EEGTransformer,
    "DeepConvNet": DeepConvNet,
}

OPTIMIZER_REGISTRY = {
    "adam": Adam,
    "adamw": AdamW,
    "SGD": SGD
}


# Following ST-EEGFormer paper (Tables F.2, F.4, F.5, F.10)
FIXED_HP = {
    "linear_probe": {  # Foundation model: frozen encoder, only train head
        "learning_rate": 5e-3,       # Table F.2(b): 0.005
        "batch_size": 64,
        "optimizer": "adamw",
        "weight_decay": 0.05,        # Table F.2(b): 0.05
        "num_epochs_cv": 2,          # Table F.2(b): 100
        "num_epochs_eval": 50,
        "warmup_epochs": 10,         # Table F.2(b): 10
        "label_smoothing": 0.1,      # Table F.2(b): 0.1
        "scheduler": "cosine",       # cosine with warmup (ST-EEGFormer default)
        "early_stopping_patience": 20,
    },
    "full": {  # Foundation model fine-tuning or baseline NN training from scratch
        "learning_rate": 5e-4,       # Table F.2(a)/F.5(a): 5e-4
        "batch_size": 64,
        "optimizer": "adamw",
        "weight_decay": 0.05,        # Table F.2(a): 0.05
        "num_epochs_cv": 100,        # Table F.2(a): 100
        "num_epochs_eval": 100,
        "warmup_epochs": 10,         # Table F.2(a): 10
        "label_smoothing": 0.1,      # Table F.2(a): 0.1
        "scheduler": "cosine",
        "early_stopping_patience": 30,
    },
    "classic_nn": {  # Classic NN models (EEGNet, Conformer, CTNet, DeepConvNet)
        "learning_rate": 3e-3,       # Table F.10(a): 3e-3
        "batch_size": 64,
        "optimizer": "adamw",
        "weight_decay": 0.05,        # Table F.10(a): 0.05
        "num_epochs_cv": 100,        # Table F.10(a): 100
        "num_epochs_eval": 100,
        "warmup_epochs": 10,         # Table F.10(a): 10
        "label_smoothing": 0.1,      # Table F.10(a): 0.1
        "scheduler": "cosine",
        "early_stopping_patience": 10,
    },
    "loo_finetune": {  # LOO Fine-Tune phase 2 (Table F.4)
        "learning_rate": 5e-5,       # Table F.4: 5e-5
        "batch_size": 32,            # Table F.4: 32
        "optimizer": "adamw",
        "weight_decay": 0.01,        # Table F.4: 0.01
        "num_epochs_cv": 50,         # Table F.4: 50
        "num_epochs_eval": 50,
        "warmup_epochs": 5,          # Table F.4: 5
        "label_smoothing": 0.1,      # Table F.4: 0.1
        "scheduler": "cosine",
        "early_stopping_patience": 10,
    },
}

# ── Per-model HP overrides for linear_probe mode ──
# Each paper has its own optimal training recipe.
# These override FIXED_HP["linear_probe"] when model_name matches.
# Keys not present here fall back to FIXED_HP["linear_probe"].
MODEL_HP_OVERRIDES = {
    "eegpt": {  # EEGPT paper: OneCycleLR, lower lr, no smoothing
        "learning_rate": 4e-4,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "scheduler": "onecycle",     # OneCycleLR stepped per batch
        "warmup_epochs": 0,          # OneCycleLR handles its own warmup via pct_start
        "num_epochs_eval": 100,
        "early_stopping_patience": 30,
    },
    "cbramod": {  # CBraMod paper: similar recipe
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "scheduler": "onecycle",
        "warmup_epochs": 0,
        "num_epochs_eval": 100,
        "early_stopping_patience": 30,
    },
    "labram": {  # LaBraM: from their released code
        "learning_rate": 4e-4,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "scheduler": "onecycle",
        "warmup_epochs": 0,
        "num_epochs_eval": 100,
        "early_stopping_patience": 30,
    },
    "biot": {
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "scheduler": "onecycle",
        "warmup_epochs": 0,
        "num_epochs_eval": 100,
        "early_stopping_patience": 30,
    },
    # steegformer: no override → uses FIXED_HP["linear_probe"] as-is (ST-EEGFormer paper settings)
    # baseline/classic_nn: no override
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    def __init__(self, model_name, model, optimizer, loss_fn, batch_size, config, early_stopper = EarlyStopper, train_data = None, val_data = None, test_data = None, training_mode = "linear_probe"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = model
        self.initial_state = deepcopy(model.state_dict())
        self.optimizer_name = optimizer
        self.loss_fn = loss_fn
        self.early_stopper = early_stopper
        self.batch_size = batch_size
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.training_mode = training_mode  # "linear_probe", "full", "classic_nn", "loo_finetune"

        # Detect if model.forward() accepts a channel_list argument
        # Foundation model (Downstream) needs it; baselines (EEGNet, Conformer, etc.) don't
        sig = inspect.signature(model.forward)
        self._model_uses_channels = "channel_list" in sig.parameters

    def _get_hp(self):
        """Get hyperparameters: FIXED_HP base merged with per-model overrides."""
        base = dict(FIXED_HP[self.training_mode])
        if self.training_mode == "linear_probe" and self.model_name in MODEL_HP_OVERRIDES:
            base.update(MODEL_HP_OVERRIDES[self.model_name])
        return base

    def _build_loss_fn(self, label_smoothing=0.0):
        """Build loss function with optional label smoothing."""
        nc = self.config["num_classes"]
        if label_smoothing > 0:
            return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return self.loss_fn

    @staticmethod
    def _build_scheduler(optimizer, num_epochs, warmup_epochs=0,
                         scheduler_type="cosine", steps_per_epoch=None, max_lr=None):
        """
        Build LR scheduler.
        - "cosine": linear warmup → cosine annealing (ST-EEGFormer default), stepped per epoch.
        - "onecycle": OneCycleLR (EEGPT/CBraMod/LaBraM), stepped per batch.
        Returns (scheduler, step_per_batch: bool).
        """
        if scheduler_type == "onecycle":
            if steps_per_epoch is None or max_lr is None:
                raise ValueError("OneCycleLR requires steps_per_epoch and max_lr")
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                pct_start=0.2,
            )
            return sched, True  # step per batch

        # Default: cosine with optional warmup (step per epoch)
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
        return sched, False  # step per epoch

    

    def build_model(self):
        "Build a new model instance"
        model_cfg = self.config["module"][self.model_name]
        model_name = model_cfg["name"]
        model_params = model_cfg.get("params", {})

        ModelClass = MODEL_REGISTRY[model_name]
        return ModelClass(**model_params)
    
    def build_optimizer(self, model, optimizer_params):
        """Build a new optimizer instance"""
        # Support both nested config format {"optimizer": {"adam": {"name": "adam"}}}
        # and direct optimizer name string (e.g. "adam")
        if "optimizer" in self.config and isinstance(self.config["optimizer"], dict):
            optimizer_cfg = self.config["optimizer"][self.optimizer_name]
            optimizer_name = optimizer_cfg["name"]
        else:
            optimizer_name = self.optimizer_name

        optimizer = OPTIMIZER_REGISTRY[optimizer_name]

        # For linear_probe, only train the classification head parameters
        if self.training_mode == "linear_probe":
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            trainable_params = model.parameters()

        return optimizer(trainable_params, **optimizer_params)

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
        
    def train_one_epoch(self, optimizer, loss_fn, model, dataloader, scheduler=None, step_per_batch=False):
        '''
        One epoch iteration of training.
        Supports two modes:
          - "linear_probe": freeze encoder, only train trainable modules
          - "full": train entire model end-to-end (baseline models)
        '''
        if self.training_mode == "linear_probe":
            # Strategy: set everything to train mode first, then set eval only
            # on children that are entirely frozen (have params but none trainable).
            # This ensures parameterless modules like Dropout in the probe head
            # stay in train mode, while frozen encoder blocks go to eval.
            model.train()
            for child in model.children():
                child_params = list(child.parameters())
                if child_params and not any(p.requires_grad for p in child_params):
                    child.eval()
        else:
            model.train()

        loss_total = 0

        for batch in tqdm(dataloader):
            # Downstream_Dataset returns (x, y, channel_id); EEGdataset returns (x, y)
            if len(batch) == 3:
                x, y, channel_list = batch
                channel_list = channel_list.to(self.device)
            else:
                x, y = batch
                channel_list = None

            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Only pass channel_list if the model actually uses it (foundation model)
            if self._model_uses_channels and channel_list is not None:
                pred = model(x, channel_list)
            else:
                pred = model(x)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # OneCycleLR steps per batch, not per epoch
            if step_per_batch and scheduler is not None:
                scheduler.step()

            loss_total = loss.item() + loss_total
        return loss_total/len(dataloader)
    

    def get_metrics(self):
        """
        Evaluation metrics — original + paper metrics:
          Original: accuracy, recall, precision, f1_score, confusion, roc_auc, kappa
          Paper:    acc1 (top-1), acc2 (top-2), bacc (balanced accuracy)
        """
        nc = self.config["num_classes"]
        metrics = {
            # ── Original metrics ──
            "accuracy":  Accuracy(task="multiclass", num_classes=nc),
            "recall":    Recall(task="multiclass", num_classes=nc, average="macro"),
            "precision": Precision(task="multiclass", num_classes=nc, average="macro"),
            "f1_score":  F1Score(task="multiclass", num_classes=nc, average="macro"),
            "confusion": ConfusionMatrix(task="multiclass", num_classes=nc),
            "roc_auc":   AUROC(task="multiclass", num_classes=nc, average="macro"),
            "kappa":     CohenKappa(task="multiclass", num_classes=nc),
            # ── Paper-specific metrics ──
            "acc1":  Accuracy(task="multiclass", num_classes=nc, top_k=1),
            "acc2":  Accuracy(task="multiclass", num_classes=nc, top_k=2),
            "bacc":  Recall(task="multiclass", num_classes=nc, average="macro"),  # balanced acc = macro recall
        }
        return metrics

    def update_metrics(self, metrics, pred, probs, y):
        """Update the torchmetric metrics"""
        # Metrics that need full probability distribution (not argmax)
        PROB_METRICS = {"roc_auc", "acc2", "auc"}
        for k, v in metrics.items():
            if k in PROB_METRICS:
                v.update(probs, y)
            else:
                v.update(pred, y)
        return self

    def compute_metrics(self, metrics):
        """Compute all metrics of evaluation"""
        return {k:v.compute() for k,v in metrics.items()}

    @staticmethod
    def _metrics_to_wandb(metrics, prefix="test"):
        """Format metrics dict for wandb logging with an optional prefix."""
        return {f"{prefix}_{k}": v for k, v in metrics.items() if k != "confusion"}

    def print_metric(self, metrics):
        """Print all evaluation metrics"""
        for k, v in metrics.items():
            if k == "confusion":
                print(f"{k}:\n{v}")
            elif hasattr(v, "item"):
                print(f"{k}: {v.item():.4f}")
            else:
                print(f"{k}: {v}")
    
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
            for batch in tqdm(dataloader):
                if len(batch) == 3:
                    x, y, channel_list = batch
                    channel_list = channel_list.to(self.device)
                else:
                    x, y = batch
                    channel_list = None

                x = x.float().to(self.device)
                y = y.long().to(self.device)

                if self._model_uses_channels and channel_list is not None:
                    pred = model(x, channel_list)
                else:
                    pred = model(x)
                probs = F.softmax(pred, dim = 1)
                pred_labels = pred.argmax(dim=1)

                #Update the metrics using the evaluation result
                self.update_metrics(metrics, pred_labels, probs, y)
                
        #Compute print all metrics
        metrics_comp = self.compute_metrics(metrics)
        self.print_metric(metrics_comp)
      
        return {k: v.item() if v.dim() == 0 else v for k, v in metrics_comp.items()}


    @staticmethod
    def _extract_labels(dataset):
        """Extract all labels from a dataset for stratification (handles Subset, ConcatDataset, etc.)."""
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            y = item[1]
            labels.append(y.item() if hasattr(y, 'item') else int(y))
        return np.array(labels)

    @time
    def tune_params_cv(self, folds, trial, eval_scheme,name_project,save = False, train_dataset = None):
        """Cross validation of the model with stratified kfold."""
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)
        all_labels = self._extract_labels(train_dataset)
        g = torch.Generator()
        g.manual_seed(42)
        # 1. Use fixed hyperparameters (no search)
        hp = self._get_hp()
        learning_rate = hp["learning_rate"]
        batch_size = hp["batch_size"]
        opt_name = hp["optimizer"]
        weight_decay = hp["weight_decay"]
        warmup_epochs = hp.get("warmup_epochs", 0)
        label_smoothing = hp.get("label_smoothing", 0.0)

        # 2. START WANDB RUN FOR THIS TRIAL
        wandb.init(
            project=eval_scheme,
            name=f"cv_{name_project}_{trial.number}",
            reinit=True,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": opt_name,
                "folds": folds,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": label_smoothing,
            }
        )

        fold_scores = []
        fold_metrics = []
        num_epochs = hp["num_epochs_cv"]

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_dataset)), all_labels)):
            early_stopper = self.early_stopper(patience=hp.get("early_stopping_patience", 10))
            print("{:^100}".format(f"---Fold_{fold}---"))

            train_subs = Subset(train_dataset, train_idx)
            val_subs = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subs, batch_size=batch_size, shuffle=True, generator = g, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_subs, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

            best = -float("inf")
            model = deepcopy(self.model).to(self.device)
            model.load_state_dict(self.initial_state)
            best_model = None

            self.optimizer_name = opt_name
            optimizer = self.build_optimizer(model, {"lr": learning_rate, "weight_decay": weight_decay})
            loss_fn = self._build_loss_fn(label_smoothing)
            scheduler, step_per_batch = self._build_scheduler(
                optimizer, num_epochs, warmup_epochs,
                scheduler_type=hp.get("scheduler", "cosine"),
                steps_per_epoch=len(train_loader),
                max_lr=learning_rate,
            )

            all_met = None

            for epoch in range(num_epochs):
                loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
                                            scheduler=scheduler, step_per_batch=step_per_batch)
                if epoch % 10 == 0:
                    print(f"Epoch: {epoch} - Loss: {loss}")

                metrics = self.predict(model=model, dataloader=val_loader)
                perf = metrics[self.config["metric"]]
                if not step_per_batch:
                    scheduler.step()

                # Log to WandB with the trial number included
                wandb.log({"acc": perf, "epoch": epoch, "fold": fold, "trial": trial.number})

                if early_stopper.should_stop(perf):
                    break

                if perf > best:
                    best_model = deepcopy(model.state_dict())
                    best = perf
                    all_met = metrics
                    
            trial.report(best, step=fold)
            if trial.should_prune():
                wandb.finish() # Safely close WandB before pruning
                raise optuna.TrialPruned()
            
            fold_scores.append(best)
            fold_metrics.append(all_met)

            if save:
                self.save_model(model, fold)

        mean_score = float(np.mean(fold_scores))
        print(f"mean fold score is {mean_score}")
        wandb.log({
            "mean_fold_score": mean_score,
            # Original metrics
            "cv_accuracy":  [met["accuracy"]  for met in fold_metrics],
            "cv_recall":    [met["recall"]    for met in fold_metrics],
            "cv_precision": [met["precision"] for met in fold_metrics],
            "cv_f1":        [met["f1_score"]  for met in fold_metrics],
            "cv_roc_auc":   [met["roc_auc"]   for met in fold_metrics],
            "cv_kappa":     [met["kappa"]     for met in fold_metrics],
            # Paper metrics
            "cv_acc1":  [met["acc1"]  for met in fold_metrics],
            "cv_acc2":  [met["acc2"]  for met in fold_metrics],
            "cv_bacc":  [met["bacc"]  for met in fold_metrics],
        })
        
        # 3. FINISH WANDB RUN
        wandb.finish()
        return mean_score


    @time
    def tune_params(self, trial, eval_scheme,name_project,train_dataset = None, val_dataset = None):
        g = torch.Generator()
        g.manual_seed(42)
        hp = self._get_hp()
        learning_rate = hp["learning_rate"]
        batch_size = hp["batch_size"]
        opt_name = hp["optimizer"]
        weight_decay = hp["weight_decay"]
        warmup_epochs = hp.get("warmup_epochs", 0)
        label_smoothing = hp.get("label_smoothing", 0.0)

        # START WANDB RUN FOR THIS TRIAL
        wandb.init(
            project=eval_scheme,
            name=f"{name_project}_{trial.number}",
            reinit=True,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": opt_name,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": label_smoothing,
            }
        )

        num_epochs = hp["num_epochs_cv"]
        early_stopper = self.early_stopper(patience=hp.get("early_stopping_patience", 10))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

        best = -float("inf")
        model = deepcopy(self.model).to(self.device)
        model.load_state_dict(self.initial_state)

        self.optimizer_name = opt_name
        optimizer = self.build_optimizer(model, {"lr": learning_rate, "weight_decay": weight_decay})
        loss_fn = self._build_loss_fn(label_smoothing)
        scheduler, step_per_batch = self._build_scheduler(
            optimizer, num_epochs, warmup_epochs,
            scheduler_type=hp.get("scheduler", "cosine"),
            steps_per_epoch=len(train_loader),
            max_lr=learning_rate,
        )

        for epoch in range(num_epochs):
            loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
                                        scheduler=scheduler, step_per_batch=step_per_batch)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")

            metrics = self.predict(model=model, dataloader=val_loader)
            perf = metrics[self.config["metric"]]

            wandb.log({"acc": perf, "epoch": epoch, "trial": trial.number})
            if not step_per_batch:
                scheduler.step()

            if early_stopper.should_stop(perf):
                break

            if perf > best:
                best = perf

            trial.report(best, step=epoch)
            if trial.should_prune():
                wandb.finish() # Safely close before pruning
                raise optuna.TrialPruned()

        # FINISH WANDB RUN
        wandb.finish()
        return best
    
    @time
    def evaluate_model(self, learning_rate, opt_name, batch_size, num_epochs, train_dataset=None, val_dataset=None, test_dataset=None):
        g = torch.Generator()
        g.manual_seed(42)
        hp = self._get_hp()
        weight_decay = hp["weight_decay"]
        warmup_epochs = hp.get("warmup_epochs", 0)
        label_smoothing = hp.get("label_smoothing", 0.0)
        patience = hp.get("early_stopping_patience", 10)

        early_stopper = self.early_stopper(patience=patience)

        # Create DataLoaders for this specific fold
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

        # Define the best model
        best = -float("inf")
        model = deepcopy(self.model).to(self.device)

        # Load the initial weights safely (in-place)
        model.load_state_dict(self.initial_state)
        best_model = model.state_dict()

        # Build the optimizer and define loss function
        self.optimizer_name = opt_name
        optimizer = self.build_optimizer(model, {"lr": learning_rate, "weight_decay": weight_decay})
        loss_fn = self._build_loss_fn(label_smoothing)

        scheduler, step_per_batch = self._build_scheduler(
            optimizer, num_epochs, warmup_epochs,
            scheduler_type=hp.get("scheduler", "cosine"),
            steps_per_epoch=len(train_loader),
            max_lr=learning_rate,
        )

        # Loop through epochs
        for epoch in range(num_epochs):
            # Train one epoch
            loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
                                        scheduler=scheduler, step_per_batch=step_per_batch)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")

            # Get the metrics for validation set
            metrics = self.predict(model=model, dataloader=val_loader)
            perf = metrics[self.config["metric"]]

            wandb.log({"acc": perf, "epoch": epoch})
            if not step_per_batch:
                scheduler.step()
            if perf > best:
                best = perf
                best_model = deepcopy(model.state_dict())
            wandb.log({"acc_eval": perf, "epoch": epoch})

            # Stop training if the val accuracy has not improved for a while
            if early_stopper.should_stop(perf):
                break
            #Report results to Optuna and prune if necessary
        model.load_state_dict(best_model)
        test_metrics = self.predict(model=model, dataloader=test_loader)
        return test_metrics
    

    def run_population(self, name_project,save=False):
        hp = self._get_hp()

        def objective(trial):
            # Only combine train and val for K-Fold CV
            train_val_dataset = ConcatDataset([self.train_data, self.val_data])
            return self.tune_params_cv(folds=5, eval_scheme = "popularion_cv",name_project=name_project,trial=trial, save=save, train_dataset=train_val_dataset)
        sampler = TPESampler(seed=42)
        #study = optuna.create_study(direction="maximize", sampler=sampler)
        #study.optimize(objective, n_trials=1)

        # --- FINAL TEST EVALUATION RUN ---
        wandb.init(
            project="population_eval",
            name=name_project,
            reinit=True,
            config={"learning_rate": hp["learning_rate"], "batch_size": hp["batch_size"],
                     "optimizer": hp["optimizer"], "weight_decay": hp["weight_decay"]}
        )
        wandb.log({"evaluation_scheme": "population"})

        metrics = self.evaluate_model(
            learning_rate=hp["learning_rate"],
            opt_name=hp["optimizer"],
            batch_size=hp["batch_size"],
            num_epochs=hp["num_epochs_eval"],
            train_dataset=self.train_data,
            val_dataset=self.val_data,
            test_dataset=self.test_data
        )

        wandb.log(self._metrics_to_wandb(metrics, prefix="test"))
        wandb.finish()
            
        
    def run_per_subject(self, name_project,participant_number, train_data_sub, val_data_sub, test_data_pop, test_data_sub, save = False):
        hp = self._get_hp()

        def objective(trial):
            train_dataset = ConcatDataset([train_data_sub, val_data_sub])
            return self.tune_params_cv(folds=5, eval_scheme = "per_subject_cv" ,name_project=name_project,trial=trial, save=save, train_dataset=train_dataset)
        sampler = TPESampler(seed=42)
        #study = optuna.create_study(direction="maximize", sampler = sampler)
        #study.optimize(objective, n_trials=1)

        # --- FINAL TEST EVALUATION RUN ---
        wandb.init(
            project="per_subject_eval_4",
            name=f"per_subject_{participant_number}_{name_project}",
            reinit=True,
            config={"learning_rate": hp["learning_rate"], "batch_size": hp["batch_size"],
                     "optimizer": hp["optimizer"], "weight_decay": hp["weight_decay"]}
        )
        wandb.log({"evaluation_scheme": "per_subject", "participant_number": participant_number})

        metrics_sub = self.evaluate_model(learning_rate=hp["learning_rate"], opt_name=hp["optimizer"], batch_size=hp["batch_size"], num_epochs=hp["num_epochs_eval"], train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_sub)

        wandb.log(self._metrics_to_wandb(metrics_sub, prefix="self"))

        #metrics_pop = self.evaluate_model(learning_rate=hp["learning_rate"], opt_name=hp["optimizer"], batch_size=hp["batch_size"], num_epochs=hp["num_epochs_eval"], train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_pop)

        #wandb.log(self._metrics_to_wandb(metrics_pop, prefix="transfer"))

       
        wandb.finish()
        
        
    def run_LOSO(self, participant_number, name_project, train_data_pop, val_data_pop, test_data_sub, save = False):
        hp = self._get_hp()

        def objective(trial):
            return self.tune_params(trial, eval_scheme="LOSO",name_project=name_project,train_dataset=train_data_pop, val_dataset=val_data_pop)
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=1)

        # --- FINAL TEST EVALUATION RUN ---
        wandb.init(
            project="LOSO",
            name=f"LOSO_{participant_number}_{name_project}",
            reinit=True,
            config={"learning_rate": hp["learning_rate"], "batch_size": hp["batch_size"],
                     "optimizer": hp["optimizer"], "weight_decay": hp["weight_decay"]}
        )
        wandb.log({"evaluation_scheme": "LOSO", "participant_number": participant_number})

        metrics_sub = self.evaluate_model(learning_rate=hp["learning_rate"], opt_name=hp["optimizer"], batch_size=hp["batch_size"], num_epochs=hp["num_epochs_eval"], train_dataset=train_data_pop, val_dataset=val_data_pop, test_dataset=test_data_sub)

        wandb.log(self._metrics_to_wandb(metrics_sub, prefix="test"))
        wandb.finish()


    @time
    def _train_model(self, learning_rate, opt_name, batch_size, num_epochs, train_dataset, val_dataset):
        """
        Train model and return the best state dict (selected by val performance).
        Used as Phase 1 of LOO Fine-Tune and LOO Drop to get population-trained weights.
        """
        g = torch.Generator()
        g.manual_seed(42)
        hp = self._get_hp()
        warmup_epochs = hp.get("warmup_epochs", 0)
        label_smoothing = hp.get("label_smoothing", 0.0)

        patience = hp.get("early_stopping_patience", 10)
        early_stopper = self.early_stopper(patience=patience)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

        best = -float("inf")
        model = deepcopy(self.model).to(self.device)
        model.load_state_dict(self.initial_state)
        best_model = deepcopy(model.state_dict())

        self.optimizer_name = opt_name
        optimizer = self.build_optimizer(model, {"lr": learning_rate, "weight_decay": hp["weight_decay"]})
        loss_fn = self._build_loss_fn(label_smoothing)
        scheduler, step_per_batch = self._build_scheduler(
            optimizer, num_epochs, warmup_epochs,
            scheduler_type=hp.get("scheduler", "cosine"),
            steps_per_epoch=len(train_loader),
            max_lr=learning_rate,
        )

        for epoch in range(num_epochs):
            loss = self.train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
                                        scheduler=scheduler, step_per_batch=step_per_batch)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")

            metrics = self.predict(model=model, dataloader=val_loader)
            perf = metrics[self.config["metric"]]
            if not step_per_batch:
                scheduler.step()

            if perf > best:
                best = perf
                best_model = deepcopy(model.state_dict())

            if early_stopper.should_stop(perf):
                break

        return best_model


    def run_LOSO_fine_tune(self, participant_number, name_project,
                           train_data_pop, val_data_pop,
                           train_data_sub, val_data_sub, test_data_sub, save=False):
        """
        LOO Fine-Tune protocol (ST-EEGFormer Protocol 5):
          Phase 1: Train on population with fixed HP
          Phase 2: Fine-tune ENTIRE model on held-out subject's data, test on subject's test set
        """
        hp = FIXED_HP[self.training_mode]

        # ── Phase 1: Population training ──
        def objective_pop(trial):
            return self.tune_params(trial, eval_scheme="LOSO_FT_pop", name_project=name_project,
                                    train_dataset=train_data_pop, val_dataset=val_data_pop)

        sampler = TPESampler(seed=42)
        study_pop = optuna.create_study(direction="maximize", sampler=sampler)
        study_pop.optimize(objective_pop, n_trials=1)

        # Train population model → get best weights
        pop_state = self._train_model(hp["learning_rate"], hp["optimizer"], hp["batch_size"], hp["num_epochs_eval"], train_data_pop, val_data_pop)

        # ── Phase 2: Fine-tune on held-out subject ──
        # Swap initial_state to population weights so evaluate_model starts from them
        original_state = self.initial_state
        original_mode = self.training_mode
        self.initial_state = pop_state
        self.training_mode = "loo_finetune"  # use LOO fine-tune specific HP (Table F.4)

        hp_ft = FIXED_HP["loo_finetune"]
        wandb.init(
            project="LOSO_fine_tune",
            name=f"LOSO_FT_{participant_number}_{name_project}",
            reinit=True,
            config={"phase": "fine_tune", "participant": participant_number,
                     "learning_rate": hp_ft["learning_rate"], "batch_size": hp_ft["batch_size"],
                     "optimizer": hp_ft["optimizer"], "weight_decay": hp_ft["weight_decay"]}
        )
        wandb.log({"evaluation_scheme": "LOSO_fine_tune", "participant_number": participant_number})

        metrics = self.evaluate_model(
            learning_rate=hp_ft["learning_rate"], opt_name=hp_ft["optimizer"], batch_size=hp_ft["batch_size"],
            num_epochs=hp_ft["num_epochs_eval"], train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_sub
        )

        wandb.log(self._metrics_to_wandb(metrics, prefix="test"))
        wandb.finish()

        # Restore original state
        self.initial_state = original_state
        self.training_mode = original_mode


    def run_LOSO_drop(self, participant_number, name_project,
                      train_data_pop, val_data_pop,
                      train_data_sub, val_data_sub, test_data_sub, save=False):
        """
        LOO Drop protocol (ST-EEGFormer Protocol 6):
          Phase 1: Train on population with fixed HP
          Phase 2: Freeze everything except classification head, train head on held-out subject's data
        """
        hp = FIXED_HP[self.training_mode]

        # ── Phase 1: Population training (identical to Fine-Tune Phase 1) ──
        def objective_pop(trial):
            return self.tune_params(trial, eval_scheme="LOSO_drop_pop", name_project=name_project,
                                    train_dataset=train_data_pop, val_dataset=val_data_pop)

        sampler = TPESampler(seed=42)
        study_pop = optuna.create_study(direction="maximize", sampler=sampler)
        study_pop.optimize(objective_pop, n_trials=1)

        # Train population model → get best weights
        pop_state = self._train_model(hp["learning_rate"], hp["optimizer"], hp["batch_size"], hp["num_epochs_eval"], train_data_pop, val_data_pop)

        # ── Phase 2: Linear probe on held-out subject ──
        original_state = self.initial_state
        original_mode = self.training_mode
        self.initial_state = pop_state
        self.training_mode = "loo_finetune"  # use LOO-specific HP for adaptation phase
        hp_ft = FIXED_HP["loo_finetune"]

        wandb.init(
            project="LOSO_drop",
            name=f"LOSO_drop_{participant_number}_{name_project}",
            reinit=True,
            config={"phase": "drop", "participant": participant_number,
                     "learning_rate": hp_ft["learning_rate"], "batch_size": hp_ft["batch_size"],
                     "optimizer": hp_ft["optimizer"], "weight_decay": hp_ft["weight_decay"]}
        )
        wandb.log({"evaluation_scheme": "LOSO_drop", "participant_number": participant_number})

        metrics = self.evaluate_model(
            learning_rate=hp_ft["learning_rate"], opt_name=hp_ft["optimizer"], batch_size=hp_ft["batch_size"],
            num_epochs=hp_ft["num_epochs_eval"], train_dataset=train_data_sub, val_dataset=val_data_sub, test_dataset=test_data_sub
        )

        wandb.log(self._metrics_to_wandb(metrics, prefix="test"))
        wandb.finish()

        # Restore original state
        self.initial_state = original_state
        self.training_mode = original_mode


    

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
    

    