# Training Pipeline

> **Relevant source files**
> * [src/build_model.py](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py)
> * [src/dataset_loaders.py](https://github.com/isblab/disobind/blob/5fffcf84/src/dataset_loaders.py)
> * [src/hparams_search.py](https://github.com/isblab/disobind/blob/5fffcf84/src/hparams_search.py)
> * [src/loss.py](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py)
> * [src/metrics.py](https://github.com/isblab/disobind/blob/5fffcf84/src/metrics.py)
> * [src/utils.py](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py)

## Purpose and Scope

This document describes the training pipeline for Disobind models, focusing on the `Trainer` class and its orchestration of the model training process. The training pipeline handles optimization, learning rate scheduling, loss calculation, metrics tracking, and model calibration. For details about the model architecture being trained, see [Epsilon_3 Network Design](/isblab/disobind/4.1-epsilon_3-network-design). For information about model configuration parameters, see [Model Configuration Files](/isblab/disobind/4.2-model-configuration-files). For practical guidance on training your own models, see [Training Your Own Models](/isblab/disobind/4.5-training-your-own-models).

---

## Trainer Class Architecture

The `Trainer` class ([src/build_model.py L25-L619](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L25-L619)

) is the central orchestrator for model training. It inherits from `nn.Module` and manages the complete training workflow including forward passes, backpropagation, optimization, validation, and calibration.

### Core Components

```mermaid
flowchart TD

Config["Training Configuration<br>from YAML"]
TrainerInit["Trainer.init()<br>src/build_model.py:26-48"]
Objective["objective: task type<br>interaction/interface"]
Optimizer["optim: AdamW/Adam/SGD"]
Scheduler["scheduler_config<br>exp/linear/multistep"]
Loss["loss_func<br>se_loss/bce"]
Metrics["num_metrics: 7<br>global averaging"]
Calibration["method: beta/platt/iso"]
Model["self.model1<br>Epsilon_3 instance"]
Opt["self.optimizer1<br>torch optimizer"]
Sched["self.scheduler1<br>LR scheduler"]
CalModel["self.cal_model<br>calibration model"]
Forward["forward()<br>lines 512-619"]
TrainStep["training_step()<br>lines 219-294"]
ValStep["validation_step()<br>lines 342-400"]
TestStep["test_step()<br>lines 404-491"]
Predict["predict()<br>lines 156-176"]
CalcLoss["calculate_loss_n_metrics()<br>lines 179-215"]
GetInputs["get_inputs()<br>lines 103-152"]
Calibrate["calibrate_model()<br>lines 298-318"]

TrainerInit --> Model
TrainerInit --> Opt
TrainerInit --> Sched
TrainerInit --> CalModel
Model --> Forward
TrainStep --> Predict
TrainStep --> CalcLoss
TrainStep --> GetInputs
TrainStep --> Calibrate

subgraph subGraph3 ["Helper Methods"]
    Predict
    CalcLoss
    GetInputs
    Calibrate
end

subgraph subGraph2 ["Training Loop Methods"]
    Forward
    TrainStep
    ValStep
    TestStep
    Forward --> TrainStep
    Forward --> ValStep
    Forward --> TestStep
end

subgraph subGraph1 ["Training Components"]
    Model
    Opt
    Sched
    CalModel
end

subgraph subGraph0 ["Trainer Class Initialization"]
    Config
    TrainerInit
    Objective
    Optimizer
    Scheduler
    Loss
    Metrics
    Calibration
    Config --> TrainerInit
    TrainerInit --> Objective
    TrainerInit --> Optimizer
    TrainerInit --> Scheduler
    TrainerInit --> Loss
    TrainerInit --> Metrics
    TrainerInit --> Calibration
end
```

**Sources:** [src/build_model.py L25-L619](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L25-L619)

### Key Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `objective` | list | Task specification: `[task_type, bin_size, pool_type, ...]` |
| `loss_func` | function | Loss function from `get_loss_function()` [src/build_model.py L38](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L38-L38) |
| `optimizer1` | torch.optim | Optimizer instance (AdamW/Adam/SGD) [src/build_model.py L50-L59](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L50-L59) |
| `scheduler1` | torch.optim.lr_scheduler | Learning rate scheduler [src/build_model.py L61-L100](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L61-L100) |
| `cal_model` | sklearn model | Calibration model (LogisticRegression/IsotonicRegression/BetaCalibration) [src/build_model.py L298-L318](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L298-L318) |
| `max_epochs` | int | Number of training epochs [src/build_model.py L43](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L43-L43) |
| `threshold` | float | Classification threshold (default 0.5) [src/build_model.py L44](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L44-L44) |
| `device` | str | 'cuda' or 'cpu' [src/build_model.py L45](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L45-L45) |

**Sources:** [src/build_model.py L26-L48](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L26-L48)

 [src/build_model.py L50-L100](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L50-L100)

---

## Training Loop

The training loop executes for `max_epochs` iterations, performing training, validation, and optionally testing at each epoch. The `HparamSearch` class [src/hparams_search.py L52](https://github.com/isblab/disobind/blob/5fffcf84/src/hparams_search.py#L52-L52)

 typically manages the execution of this loop via the `Trainer.forward()` method [src/build_model.py L512](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L512-L512)

### Main Training Flow

```mermaid
flowchart TD

Start["Trainer.forward() Entry<br>src/build_model.py:512"]
Initialize["Initialize<br>optimizer, scheduler<br>lines 541-543"]
EpochLoop["For each<br>epoch"]
TrainMode["model.train()<br>line 548"]
TrainStep["training_step(train_set, epoch)<br>lines 219-294"]
EvalMode["model.eval()<br>line 551"]
ValStep["validation_step(dev_set, epoch)<br>lines 342-400"]
RecordMetrics["Record epoch metrics<br>lines 558-567"]
PrintMetrics["Print Loss, Recall,<br>Precision, F1<br>lines 571-576"]
TestFinal["test_step(test_set)<br>lines 585"]
SaveResults["Return model,<br>cal_model, metrics<br>line 618"]
End["Training Complete"]

Start --> Initialize
Initialize --> EpochLoop
EpochLoop --> TrainMode
TrainMode --> TrainStep
TrainStep --> EvalMode
EvalMode --> ValStep
ValStep --> RecordMetrics
RecordMetrics --> PrintMetrics
PrintMetrics --> EpochLoop
EpochLoop --> TestFinal
TestFinal --> SaveResults
SaveResults --> End
```

**Sources:** [src/build_model.py L512-L619](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L512-L619)

 [src/hparams_search.py L173-L220](https://github.com/isblab/disobind/blob/5fffcf84/src/hparams_search.py#L173-L220)

### Training Step Details

The `training_step()` method processes all mini-batches in the training set provided by the `DatasetLoader` [src/dataset_loaders.py L69](https://github.com/isblab/disobind/blob/5fffcf84/src/dataset_loaders.py#L69-L69)

:

```mermaid
flowchart TD

StartTrain["training_step(train_set, epoch)<br>line 219"]
InitBatch["Initialize batch_dict<br>lines 236-237"]
BatchLoop["For each<br>batch in<br>train_set"]
ZeroGrad["optimizer.zero_grad()<br>line 244"]
Predict["predict(batch, train=True)<br>line 246"]
CalcLoss["calculate_loss_n_metrics()<br>line 248"]
RecordBatch["Record batch metrics<br>lines 250-257"]
Backward["loss.backward()<br>line 260"]
ClipGrad["max_norm<br>!= None?"]
Clip["torch.nn.utils.clip_grad_norm()<br>line 273"]
OptStep["optimizer.step()<br>line 276"]
SWACheck["Using SWA<br>scheduler?"]
UpdateSWA["Update SWA model<br>lines 278-280"]
CalibrateFinal["calibrate_model()<br>lines 287-291"]
SchedulerStep["scheduler.step()<br>lines 282-283"]
ReturnMetrics["Return batch_dict<br>line 294"]

StartTrain --> InitBatch
InitBatch --> BatchLoop
BatchLoop --> ZeroGrad
ZeroGrad --> Predict
Predict --> CalcLoss
CalcLoss --> RecordBatch
RecordBatch --> Backward
Backward --> ClipGrad
ClipGrad --> Clip
ClipGrad --> OptStep
Clip --> OptStep
OptStep --> SWACheck
SWACheck --> UpdateSWA
SWACheck --> BatchLoop
UpdateSWA --> BatchLoop
BatchLoop --> CalibrateFinal
BatchLoop --> SchedulerStep
SchedulerStep --> ReturnMetrics
CalibrateFinal --> ReturnMetrics
```

**Sources:** [src/build_model.py L219-L294](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L219-L294)

 [src/dataset_loaders.py L106-L114](https://github.com/isblab/disobind/blob/5fffcf84/src/dataset_loaders.py#L106-L114)

---

## Input Preparation and Forward Pass

Before each forward pass, inputs must be prepared according to the training objective.

### Input Processing Pipeline

```mermaid
flowchart TD

GetInputs["Trainer.get_inputs(batch)<br>src/build_model.py:103"]
SplitBatch["Split batch into<br>target, p1_emb, p2_emb<br>lines 125-136"]
PrepInput["prepare_input()<br>src/utils.py lines 92-205"]
CheckObj["Objective<br>type?"]
MaxPool2D["Apply MaxPool2d for<br>coarse-graining<br>lines 144-149"]
InterfaceProj["Project to interface<br>representation<br>lines 169-193"]
ToDevice["Move to device<br>lines 146-150"]
ReturnTensors["Return:<br>p1_emb, p2_emb,<br>target, masks"]
PredictForward["Trainer.predict(batch)<br>src/build_model.py:156"]
ModelForward["model(p1_emb, p2_emb,<br>interaction_mask)<br>line 173"]
ReturnPreds["Return predictions,<br>target, mask<br>line 176"]

GetInputs --> SplitBatch
SplitBatch --> PrepInput
PrepInput --> CheckObj
CheckObj --> MaxPool2D
CheckObj --> InterfaceProj
MaxPool2D --> ToDevice
InterfaceProj --> ToDevice
ToDevice --> ReturnTensors
ReturnTensors --> PredictForward
PredictForward --> ModelForward
ModelForward --> ReturnPreds
```

**Sources:** [src/build_model.py L103-L176](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L103-L176)

 [src/utils.py L92-L205](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L92-L205)

### Coarse-Graining Process

For binned (coarse-grained) predictions, the input preparation in `prepare_input` [src/utils.py L92](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L92-L92)

 applies pooling operations:

| Objective | CG Level | Pooling Type | Implementation |
| --- | --- | --- | --- |
| `interaction_bin` | > 1 | `MaxPool2d` | `nn.MaxPool2d(kernel_size=bin_size, stride=bin_size)` [src/utils.py L144](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L144-L144) |
| `interface_bin` | > 1 | `MaxPool2d` | Applied to projected interface mask [src/utils.py L153](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L153-L153) |
| Embedding Averaging | > 1 | `AvgPool1d` | `nn.AvgPool1d` if `bin_input` is True [src/utils.py L136](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L136-L136) |

**Sources:** [src/utils.py L129-L156](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L129-L156)

---

## Optimization Strategy

### Optimizer Configuration

The `Trainer` supports three optimizers initialized in the `optimizer()` method [src/build_model.py L50](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L50-L50)

:

```mermaid
flowchart TD

OptimizerMethod["Trainer.optimizer()<br>src/build_model.py:50"]
CheckType["optimizer<br>type?"]
AdamOpt["torch.optim.Adam<br>lr, weight_decay, amsgrad<br>line 52"]
SGDOpt["torch.optim.SGD<br>lr, weight_decay, momentum=0.9<br>line 55"]
AdamWOpt["torch.optim.AdamW<br>lr, weight_decay, amsgrad<br>line 58"]
Return["Return optimizer"]

OptimizerMethod --> CheckType
CheckType --> AdamOpt
CheckType --> SGDOpt
CheckType --> AdamWOpt
AdamOpt --> Return
SGDOpt --> Return
AdamWOpt --> Return
```

**Sources:** [src/build_model.py L50-L59](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L50-L59)

### Learning Rate Schedulers

The `scheduler()` method ([src/build_model.py L61-L100](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L61-L100)

) configures various learning rate schedules based on the `scheduler_config` [src/build_model.py L33](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L33-L33)

:

| Scheduler | Type | Key Parameters |
| --- | --- | --- |
| `swa` | Stochastic Weight Averaging | `swa_lr`, `swa_start` [src/build_model.py L64-L68](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L64-L68) |
| `cycliclr` | Cyclic Learning Rate | `base_lr`, `max_lr`, `step_size_up` [src/build_model.py L69-L75](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L69-L75) |
| `multistep` | MultiStep Decay | `milestones`, `gamma` [src/build_model.py L76-L80](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L76-L80) |
| `exp` | Exponential Decay | `gamma` [src/build_model.py L82-L85](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L82-L85) |
| `linear` | Linear Decay | `start_factor`, `end_factor`, `total_iters` [src/build_model.py L87-L94](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L87-L94) |

**Sources:** [src/build_model.py L61-L100](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L61-L100)

---

## Loss Functions

Disobind implements several loss functions in `src/loss.py` to handle the specific requirements of protein interaction prediction.

| Loss Class | Implementation | Use Case |
| --- | --- | --- |
| `BCE` | `nn.BCELoss` | Standard binary cross entropy [src/loss.py L55](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L55-L55) |
| `BCEwithLogits` | `nn.BCEWithLogitsLoss` | Weighted BCE with logit input [src/loss.py L104](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L104-L104) |
| `FocalLoss` | Custom implementation | Handles class imbalance by focusing on hard examples [src/loss.py L135](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L135-L135) |
| `SingularityEnhancedLoss` | Custom implementation | Specifically designed for contact prediction [src/loss.py L162](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L162-L162) |
| `InterfaceLoss` | Wrapper | Combines losses for dual interface predictions [src/loss.py L189](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L189-L189) |

The `get_loss_function()` helper in `src/loss.py` (referenced in [src/build_model.py L38](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L38-L38)

) instantiates these classes based on the configuration.

**Sources:** [src/loss.py L1-L205](https://github.com/isblab/disobind/blob/5fffcf84/src/loss.py#L1-L205)

 [src/build_model.py L38](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L38-L38)

---

## Validation and Testing

### Validation Step

The `validation_step()` method [src/build_model.py L342](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L342-L342)

 evaluates the model on the development set without gradient updates. It also collects predictions for calibration during the final epoch [src/build_model.py L371-L379](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L371-L379)

### Test Step

The `test_step()` method [src/build_model.py L404](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L404-L404)

 generates final predictions on the test set and produces reliability diagrams using `plot_reliabity_diagram` [src/utils.py L21](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L21-L21)

```mermaid
flowchart TD

TestStart["test_step(test_set, file_name)<br>line 404"]
NoGradTest["with torch.no_grad():<br>line 428"]
TestLoop["For each<br>batch"]
PredTest["predict(batch)<br>line 432"]
GetCalPreds["get_calibrated_preds()<br>lines 447-454"]
CalcMetricsTest["calculate_loss_n_metrics()<br>line 460"]
RecordMetricsTest["Record metrics<br>lines 463-469"]
AvgMetrics["Average metrics<br>line 474"]
CheckMethod["Calibration<br>method?"]
SkipPlot["Skip reliability plot<br>line 481"]
PlotReliability["plot_reliabity_diagram()<br>src/utils.py:21"]
PrintResults["Print Recall, Precision, F1<br>lines 488-489"]
Return["Return test_met,<br>test_pred, test_target<br>line 491"]

TestStart --> NoGradTest
NoGradTest --> TestLoop
TestLoop --> PredTest
PredTest --> GetCalPreds
GetCalPreds --> CalcMetricsTest
CalcMetricsTest --> RecordMetricsTest
RecordMetricsTest --> TestLoop
TestLoop --> AvgMetrics
AvgMetrics --> CheckMethod
CheckMethod --> SkipPlot
CheckMethod --> PlotReliability
SkipPlot --> PrintResults
PlotReliability --> PrintResults
PrintResults --> Return
```

**Sources:** [src/build_model.py L404-L491](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L404-L491)

 [src/utils.py L21-L62](https://github.com/isblab/disobind/blob/5fffcf84/src/utils.py#L21-L62)

---

## Model Calibration

Calibration improves the reliability of predicted probabilities. The `Trainer` supports multiple calibration methods via `calibrate_model` [src/build_model.py L298](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L298-L298)

### Calibration Methods Comparison

| Method | Implementation | Library |
| --- | --- | --- |
| `platt` | `LogisticRegression` | `sklearn.linear_model` [src/build_model.py L302](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L302-L302) |
| `iso` | `IsotonicRegression` | `sklearn.isotonic` [src/build_model.py L307](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L307-L307) |
| `beta-abm` | `BetaCalibration` | `betacal` [src/build_model.py L311](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L311-L311) |
| `temp` | Temperature Scaling | Internal [src/build_model.py L318](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L318-L318) |

Calibrated predictions are retrieved using `get_calibrated_preds()` [src/build_model.py L322](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L322-L322)

 which applies the fitted `self.cal_model`.

**Sources:** [src/build_model.py L298-L338](https://github.com/isblab/disobind/blob/5fffcf84/src/build_model.py#L298-L338)