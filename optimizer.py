import torch
from torch.optim import Adam, AdamW

def build_optimizer(model, config):
    lr = config.model.learning_rate
    weight_decay = config.model.weight_decay
    optimizer_name = config.model.optimizer
    
    if optimizer_name == "Adam":
        optimizer = Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if optimizer_name == "AdamW":
        # 為 AdamW 分組參數
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            amsgrad=True
        )
    return optimizer