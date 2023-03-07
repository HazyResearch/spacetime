import torch
from tqdm import tqdm


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
        try: model.set_eval()
        except: model.eval()
        grad_enabled = False
        
        
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
            y_t = torch.cat([x, y], dim=1)  # supervise all time-steps
            
            # Closed-loop supervision
            w0, w1, w2 = config.criterion_weights
            # loss = w0 * criterions[config.loss](y_c[:, model.kernel_dim:-1, :], 
            #                                     y_t[:, model.kernel_dim+1:, :].to(config.device))
            loss = w0 * criterions[config.loss](y_c[:, model.lag-1:-1, :], 
                                                y_t[:, model.lag:, :].to(config.device))
            # breakpoint()
            if not model.inference_only:
                # Open-loop output supervision,
                # -> Offset by 1 bc next time-step prediction
                loss += w1 * criterions[config.loss](y_o[:, model.kernel_dim:-1, :],
                                                     x[:, model.kernel_dim + 1:, :].to(config.device))
                # breakpoint()
                # Closed-loop next-time-step input supervision
                # -> Offset by 1 bc next time-step prediction
                z_c, z_t = z_pred
                z_c = z_c[:, :model.lag, :]
                loss += w2 * criterions[config.loss](z_c[:, model.kernel_dim:-1, :],
                                                     z_t[:, model.kernel_dim+1:, :])
                # breakpoint()
                
            if grad_enabled:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save metrics
            y_c = y_c.detach().cpu()
            y_t = y_t.detach().cpu()
            u = u.detach().cpu()
            
            for k, criterion in criterions.items():
                if k == 'correct':
                    metrics[k] += criterion(y_c[:, model.lag-1:-1, :], 
                                            y_t[:, model.lag:, :]).sum().item()
                else:
                    metrics[k] += (criterion(y_c, y_t).mean() * len(y_t)).item()
            metrics['total'] += len(y)
            
            description = f'└── {split} batch {int(batch_ix)}/{len(pbar)}'
            for metric_name, metric in metrics.items():
                if metric_name == 'correct':
                    description += f' | {metric_name} (acc. %): {int(metric):>5d}/{int(metrics["total"])} = {metric / metrics["total"] * 100:.3f}%'
                elif metric_name != 'total':
                    description += f' | {metric_name}: {metric / metrics["total"]:.3f}'
            pbar.set_description(description)
            
            try:
                y_o = y_o.detach().cpu()
            except: 
                pass
            
    model.cpu()
    return model, metrics