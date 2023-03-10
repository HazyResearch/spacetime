import torch
from tqdm import tqdm

from loss import get_loss


def compute_informer_metrics(y_pred, y_true):
    metrics = {}
    criterions = {f'informer_{name}': get_loss(f'informer_{name}') 
                  for name in ['rmse', 'mse', 'mae']}
    for k, criterion in metrics:
        metrics[k] = criterion(y_pred, y_true)
    return metrics

def shared_step(model, dataloader, optimizer, scheduler, criterions, epoch, 
                config, split, input_transform=None, output_transform=None):

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
        try: 
            model.set_eval()
        except Exception as e: 
            print(e)
            model.eval()
        grad_enabled = False
        
    # Save predictions
    total_y_true = []
    total_y_pred = []
    total_y_true_informer = []
    total_y_pred_informer = []
        
    with torch.set_grad_enabled(grad_enabled):
        
        pbar = tqdm(dataloader, leave=False)
        model.to(config.device)
        
        for batch_ix, data in enumerate(pbar):
            x, y, *z = data
            
            # Only take in lag terms for input
            x = x[:, :model.lag, :]
            
            # Transform batch data 
            u = input_transform(x)
            u = u.to(config.device)

            # Return (model outputs), (model last-layer next-step inputs)
            y_pred, z_pred = model(u)
            # (y_c, y_o) = y_pred
            # (z_u_pred, z_u_true) = z_pred
            y_pred = [output_transform(_y) if _y is not None else _y 
                      for _y in y_pred]
            y_c, y_o = y_pred 
            
            # y_c = y_c[:, model.lag:, :]
            # y_c = torch.cat([x[:, :-1, :].to(y_c.device), y_c], dim=1)
            y_t = torch.cat([x[:, :, :], y], dim=1)  # supervise all time-steps
            
            # Closed-loop supervision
            w0, w1, w2 = config.criterion_weights
            if model.replicate == 1:
                loss = torch.mean(w0 * criterions[config.loss](y_c[:, model.kernel_dim-1:-1, :], 
                                                               y_t[:, model.kernel_dim:, :].to(config.device)))
            else:
                # loss = torch.mean(w0 * criterions[config.loss](y_c[:, model.lag-2:-1, :], 
                #                                                y_t[:, model.lag-1:, :].to(config.device)))
                loss = torch.mean(w0 * criterions[config.loss](y_c,  # [:, model.lag-1:-1, :], 
                                                               y_t[:, model.lag:, :].to(config.device)))
                
            # breakpoint()
            if not model.inference_only:
                # Open-loop output supervision,
                # -> Offset by 1 bc next time-step prediction
                loss += torch.mean(w1 * criterions[config.loss](y_o[:, model.kernel_dim-1:, :],
                                                                y_t[:, model.kernel_dim:model.lag+1, :].to(config.device)))
                    
                    # criterions[config.loss](y_o[:, model.kernel_dim:-1, :], x[:, model.kernel_dim + 1:, :].to(config.device))
                # Closed-loop next-time-step input supervision
                # -> Offset by 1 bc next time-step prediction
                # z_c, z_t = z_pred
                # z_c = z_c[:, :model.lag, :]
                # loss += torch.mean(w2 * criterions[config.loss](z_c[:, model.kernel_dim-1:-1, :],
                #                                                 z_t[:, model.kernel_dim:, :]))
#                 # breakpoint()
                
            if grad_enabled:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save metrics
            y_c = y_c.detach().cpu()
            y_t = y_t.detach().cpu()
            u = u.detach().cpu()
            
            # Compute metrics only for horizon terms
            # y_c_horizon = y_c[:, model.lag-2:-1, :]
            y_c_horizon = y_c
            # y_t_horizon = y_t[:, model.lag-1:, :]
            y_t_horizon = y_t[:, model.lag:, :]
            
            for k, criterion in criterions.items():
                if k == 'correct':
                    metrics[k] += criterion(y_c_horizon, 
                                            y_t_horizon).sum().item()
                else:
                    metrics[k] += criterion(y_c_horizon, 
                                            y_t_horizon).sum().item()
            metrics['total'] += (y_t_horizon.shape[0] * y_t_horizon.shape[1])  # Batch size * horizon
            
            description = f'└── {split} batch {int(batch_ix)}/{len(pbar)}'
            for metric_name, metric in metrics.items():
                if metric_name == 'correct':
                    description += f' | {metric_name} (acc. %): {int(metric):>5d}/{int(metrics["total"])} = {metric / metrics["total"] * 100:.3f}%'
                elif 'informer' in metric_name:
                    description += f' | {metric_name}: {metric / (batch_ix + 1):.3f}'
                elif metric_name != 'total':
                    description += f' | {metric_name}: {metric / metrics["total"]:.3f}'
            pbar.set_description(description)
            
            try:
                y_o = y_o.detach().cpu()
            except: 
                pass
            
            # Save these for Informer metrics (standardized)
            total_y_pred_informer.append(y_c_horizon)
            total_y_true_informer.append(y_t_horizon)  
            
            # Now save raw-scale metrics
            total_y_true.append(dataloader.dataset.inverse_transform(y_c_horizon))
            total_y_pred.append(dataloader.dataset.inverse_transform(y_t_horizon))
                                    
    total_y_true_informer = torch.cat(total_y_true_informer, dim=0)
    total_y_pred_informer = torch.cat(total_y_pred_informer, dim=0)
    
    total_y = {'true': torch.cat(total_y_true, dim=0),
               'pred': torch.cat(total_y_pred, dim=0),
               'true_informer': total_y_true_informer,
               'pred_informer': total_y_pred_informer}
    
    informer_metrics = compute_informer_metrics(total_y_true_informer,
                                                total_y_pred_informer)
    for k, v in informer_metrics.items():
        metrics[f'no_reduce_{k}'] = v
        
    for k, metric in metrics.items():
        if k != 'total' and 'informer' not in k:
            metrics[k] = metric / metrics['total']
        elif 'informer' in k and 'no_reduce' not in k:
            metrics[k] = metric / (batch_ix + 1)
        else:
            metrics[k] = metric
            
    model.cpu()
    return model, metrics, total_y 