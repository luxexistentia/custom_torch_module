import torch
from timm.data import Mixup
from custom_torch_module import setup_utils
from tqdm.auto import tqdm
from torcheval.metrics import MulticlassF1Score
from statistics import harmonic_mean
import copy

class Model_Trainer():
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, loss_fn, device, trainable_part, model_classifier, num_classes, weights=None, scheduler=None, label_smoothing=None, eval_func=MulticlassF1Score(), use_amp=True):
        self.device = device
        torch.set_default_device(self.device)
    
        self.model = model
        self.train_dataloader, self.test_dataloader = train_dataloader, test_dataloader
        self.optimizer, self.loss_fn = optimizer, loss_fn
        
        self.trainable_part, self.model_classifier = trainable_part, model_classifier
        self.label_smoothing = label_smoothing
        self.eval_func = eval_func

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        if weights:
            self.load_weights(weights)
        else:
            self.weights = None

        self.scheduler = scheduler if scheduler else None
        # MixUp
        if self.label_smoothing:
            self.mixup_fn = Mixup(
                mixup_alpha=(1-self.label_smoothing),  # alpha 값 설정
                cutmix_alpha=0.0,  # CutMix를 사용하지 않을 경우 0으로 설정
                label_smoothing=self.label_smoothing,
                num_classes=num_classes)

        print("[info] Sucessfully created the instance.")
        print(f"[info] Device : {self.device} | Trainable part : {self.trainable_part}")

    def train_step(self):
        self.model.train()
        
        train_loss = []
        train_logits = torch.empty([0])
        train_labels = torch.empty([0])
        self.eval_func.reset()
        for X, y in self.train_dataloader:
            with torch.autocast(device_type=self.device, enabled=self.use_amp):# auto Cast to torch.float16 if self.use_amp is True
                y_hard_label = y.to(self.device)
                
                if self.label_smoothing:
                    X, y = self.mixup_fn(X, y)
                    
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                
                loss = self.loss_fn(y_logits, y)
    
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss.append(loss.item())
            self.eval_func.update(y_logits, y_hard_label)
        train_loss = harmonic_mean(train_loss)
        train_eval_score = self.eval_func.compute()
        
        return train_loss, train_eval_score
    
    def test_step(self):
        self.model.eval()
        
        test_loss = []
        test_logits = torch.empty([0])
        test_labels = torch.empty([0])
        self.eval_func.reset()
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
    
                loss = self.loss_fn(y_logits, y)
    
                test_loss.append(loss.item())
                self.eval_func.update(y_logits, y)
        test_loss = harmonic_mean(test_loss)
        test_eval_score = self.eval_func.compute()
    
        return test_loss, test_eval_score
    def train_epochs(self, epochs):
        best_eval_score = -100000
        best_loss = 100000
        for epoch in tqdm(range(1, epochs+1)):
            print("=-"*25)
            print(f"[info] Epoch{epoch} | Device : {self.device} | Trainable part : {self.trainable_part}\n")
            
            #train epoch
            train_loss, train_eval_score = self.train_step()
            print(f"Train Loss:{train_loss:.4f} | Train eval Score:{train_eval_score:.4f}")
            
            #test epoch
            test_loss, test_eval_score = self.test_step()
            print(f"Test Loss:{test_loss:.4f} | Test eval Score:{test_eval_score:.4f}")

            if self.scheduler:
                self.scheduler.step()
        
            loss_ratio = train_loss / test_loss
            eval_score_ratio = test_eval_score / train_eval_score
            
            print(f"Loss Ratio:{loss_ratio:.4f} | eval Score Ratio:{eval_score_ratio:.4f}\n")
        
            if best_eval_score < test_eval_score or (best_loss > test_loss and best_eval_score == test_eval_score):
                best_weights = copy.deepcopy(self.model.state_dict())
                
                best_eval_score = test_eval_score
                best_loss = test_loss
                best_epoch = epoch
            print(f"Best model(from {best_epoch}epoch) | eval Score:{best_eval_score:.4f} | Loss:{best_loss:.4f}")
        return best_weights, best_eval_score, best_loss
    
    def train(self, epochs, weights=None, file_name=None, save_weights=True):
        if weights != None:
            self.load_weights(weights)
        elif self.trainable_part != 0 and self.weights == None:
            print("[info] No Weights were found for the model. Freezing the model & Starting train of the head part first...")
            
            self.model = setup_utils.freeze_model(self.model, 0, self.model_classifier) # 웨이트 없으면 먼저 헤드 훈련 진행 후 모델 전체 훈련 진행
            best_weights, best_eval_score, best_loss = self.train_epochs(epochs)
            
            print("[info] Pretrain of head part has done. Loading the best weights...")
            print(f"[info] Best eval Score:{best_eval_score:.4f} | Best Loss:{best_loss:.6f}")
    
            self.model.load_state_dict(best_weights)
            print("[info] Succesfully loaded the weights")
    
        self.model = setup_utils.freeze_model(self.model, self.trainable_part, self.model_classifier) # need to be edited
        
        best_weights, best_eval_score, best_loss = self.train_epochs(epochs)
    
        if save_weights:
            file_name = file_name + f" (eval Score {best_eval_score:.4f}, Loss {best_loss:6f}).pth"
            torch.save(obj=best_weights, f=file_name)
            print("[info] Best weights has saved.")
            print(f"[info] Best eval score:{best_eval_score:.4f} | Best Loss:{best_loss:.6f}")

        self.load_weights(best_weights)
        print("[info] Best weights has loaded to the model.")
    
    def load_weights(self, weights):
        print("[info] Loading the weights to the model...")
        
        self.weights = weights
        self.model.load_state_dict(weights)

        print("[info] Succesfully loaded the weights")