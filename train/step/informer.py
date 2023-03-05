import torch
from tqdm import tqdm


def shared_step(model, dataloader, optimizer, scheduler, criterions, 
                epoch, device, split, input_transform=None, output_transform=None):

    if input_transform is None:
        input_transform = lambda x: x
        
    if output_transform is None:
        output_transform = lambda y: y

    # Save step-wise metrics
    metrics = {'total': 0.}
    for k in criterions.keys():
        metrics[k] = 0. 
        
    if split == 'train':
        try: model.set_train()
        except: model.train()
        model.zero_grad()
        grad_enabled = True
    else:
        try: model.set_eval()
        except: model.eval()
        grad_enabled = False
        
        
    with torch.set_grad_enabled(grad_enabled):
        
        pbar = tqdm(dataloader, leave=False)
        model.to(config.device)
        
        for batch_ix, data in enumerate(pbar):
            u, y, *z = data
            
            # Transform batch data 
            u = input_transform(x)
            u = u.to(config.device)

            # Return (model outputs), (model last-layer next-step inputs)
            y_pred, z_pred = model(u)
            y_pred = [output_transform(_y) for _y in y_pred if _y is not None]
            
            y_true = y
            loss = criterions['loss'](y_pred, y_true.to(config.device))
            
            if grad_enabled:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save metrics
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()
            u = u.detach().cpu()
            
            for k, criterion in criterions.items():
                if k == 'correct':
                    metrics[k] += criterion(y_pred, y_true).sum().item()
                else:
                    metrics[k] += (criterion(y_pred, y_true).mean() * len(y_true)).item()
            metrics['total'] += len(y)
            
            description = f'└── {split} batch {int(batch_ix)}/{len(pbar)}'
            for metric_name, metric in metrics.items():
                if metric_name == 'correct':
                    description += f' | {metric_name} (acc. %): {int(metric):>5d}/{int(metrics["total"])} = {metric / metrics["total"] * 100:.3f}%'
                elif metric_name != 'total':
                    description += f' | {metric_name}: {metric / metrics["total"]:.3f}'
            pbar.set_description(description)
    metrics['acc'] = metrics['correct'] / metrics['total'] * 100.
    model.cpu()
    return model, metrics