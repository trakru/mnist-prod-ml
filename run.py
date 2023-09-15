import optuna
from model.train import objective

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    print(f'Number of finished trials: {len(study.trials)}')
    print(f'Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print(f'  Params: {trial.params}')
