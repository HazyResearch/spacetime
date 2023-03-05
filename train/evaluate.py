from .epoch import run_epoch


def evaluate_model(model, **kwargs):
    model.eval()
    log_metrics = {}
    with torch.no_grad():
        _, metrics = run_epoch(model, **kwargs)
        if kwargs['wandb'] is not None:
            for split in metrics.keys():
                for k, v in metrics[split].items():
                    log_metrics[f'eval_{split}/{k}'] = v
            kwargs['wandb'].log(log_metrics)
        # Print out evaluation metrics
        for split in metrics.keys():
            print('-'*4, f'Eval {split}', '-'*4)
            for k, v in metrics[split].items():
                print(f'- {k}: {v}')
                log_metrics[f'eval_{split}/{k}'] = v
    return model, log_metrics