import torch
from timm.data import Mixup
from custom_torch_module import setup_utils
from tqdm.auto import tqdm
from torcheval.metrics.functional import multiclass_f1_score
from statistics import harmonic_mean

class Model_Trainer():
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, loss_fn, device, trainable_part, model_classifier, weights=None, scheduler=None, label_smoothing=None):
        self.device = device
        torch.set_default_device(self.device)
    
        self.model = model
        self.train_dataloader, self.test_dataloader = train_dataloader, test_dataloader
        self.optimizer, self.loss_fn = optimizer, loss_fn
        
        self.trainable_part, self.model_classifier = trainable_part, model_classifier
        self.label_smoothing = label_smoothing
        
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
                num_classes=model_classifier.out_features)

        print("[info] Sucessfully created the instance.")

    def train_step(self):
        self.model.train()
        train_loss = []
        train_logits = torch.empty([0])
        train_labels = torch.empty([0])
        for X, y in self.train_dataloader:
            y_hard_label = y.to(self.device)
            
            if self.label_smoothing:
                X, y = self.mixup_fn(X, y)
                
            X, y = X.to(self.device), y.to(self.device)
            
            y_logits = self.model(X)
    
            self.optimizer.zero_grad()
            loss = self.loss_fn(y_logits, y)
            loss.backward()
            self.optimizer.step()
    
            train_loss.append(loss.item())
            train_logits = torch.cat((train_logits, y_logits.argmax(dim=1)))
            train_labels = torch.cat((train_labels, y_hard_label))
        train_loss = harmonic_mean(train_loss)
        train_f1_score = multiclass_f1_score(train_logits, train_labels)
        
        return train_loss, train_f1_score
    
    def test_step(self):
        self.model.eval()
        
        test_loss = []
        test_logits = torch.empty([0])
        test_labels = torch.empty([0])
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
    
                loss = self.loss_fn(y_logits, y)
    
                test_loss.append(loss.item())
                test_logits = torch.cat((test_logits, y_logits.argmax(dim=1)))
                test_labels = torch.cat((test_labels, y))
        test_loss = harmonic_mean(test_loss)
        test_f1_score = multiclass_f1_score(test_logits, test_labels)
    
        return test_loss, test_f1_score
    def train_epochs(self, epochs):
        best_f1_score = 0
        best_loss = 100000
        for epoch in tqdm(range(1, epochs+1)):
            print("=-"*25)
            print(f"Epoch{epoch}")
            
            #train epoch
            train_loss, train_f1_score = self.train_step()
            print(f"Train Loss:{train_loss:.4f} | Train F1 Score:{100*train_f1_score:.2f}%")
            
            #test epoch
            test_loss, test_f1_score = self.test_step()
            print(f"Test Loss:{test_loss:.4f} | Test F1 Score:{100*test_f1_score:.2f}%")

            if self.scheduler:
                self.scheduler.step()
        
            loss_ratio = train_loss / test_loss
            f1_score_ratio = test_f1_score / train_f1_score
            
            print(f"Loss Ratio:{loss_ratio:.4f} | F1 Score Ratio:{f1_score_ratio:.4f}")
        
            if best_f1_score < test_f1_score or (best_loss > test_loss and best_f1_score == test_f1_score):
                best_weights = self.model.state_dict()
                
                best_f1_score = test_f1_score
                best_loss = test_loss
            print(f"Best model | F1 Score:{best_f1_score*100:.2f}% | Loss:{best_loss:.4f}")
        return best_weights, best_f1_score, best_loss
    
    def train(self, epochs, weights=None, file_name=None, save_weights=True):
        if weights != None:
            self.load_weights(weights)
        elif self.trainable_part != 0 and self.weights == None:
            print("[info] No Weights were found for the model. Freezing the model & Starting train of the head part first...")
            
            self.model = setup_utils.freeze_model(self.model, 0, self.model_classifier) # 웨이트 없으면 먼저 헤드 훈련 진행 후 모델 전체 훈련 진행
            best_weights, best_f1_score, best_loss = self.train_epochs(epochs)
            
            print("[info] Pretrain of head part has done. Loading the best weights...")
            print(f"[info] Best F1 Score:{best_f1_score*100:.2f}% | Best Loss:{best_loss:.6f}")
    
            self.model.load_state_dict(best_weights)
            print("[info] Succesfully loaded the weights")
    
        self.model = setup_utils.freeze_model(self.model, self.trainable_part, self.model_classifier) # need to be edited
        
        best_weights, best_f1_score, best_loss = self.train_epochs(epochs)
    
        if save_weights:
            file_name = file_name + f" (F1 Score {best_f1_score*100:.2f}%, Loss {best_loss:6f}).pth"
            torch.save(obj=best_weights, f=file_name)
            print("[info] Best weights has saved.")
            print(f"[info] Best F1 score:{best_f1_score*100:.2f}% | Best Loss:{best_loss:.6f}")

            self.load_weights(best_weights)
    
    def load_weights(self, weights):
        print("[info] Loading the weights to the model...")
        
        self.weights = weights
        self.model.load_state_dict(weights)

        print("[info] Succesfully loaded the weights")