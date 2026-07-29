# Project Milestones tracking Registry

Progress log tracking the development milestones of the Lightweight Image Super-Resolution framework.

---

## Completed Phases

### Phase 1: Repository Infrastructure ✅
* **Description**: Setup repository directory layout, dependencies environment, logging utilities, and basic configurations system.
* **Deliverables**:
  * Unified logging `logger.py`
  * Reproducibility controls `seed.py`
  * Compute backend diagnostics `device.py`
  * Path managers `paths.py`

### Phase 2: Configuration System ✅
* **Description**: Establish modular YAML configurations parsing, strongly-typed schema validations, and run settings backups.
* **Deliverables**:
  * Dataclass-based configurations parsing `config.py`
  * Default configurations mapping files

### Phase 3: Dataset Pipeline ✅
* **Description**: Create spatial crops, augmentations, and dynamic degradation pipelines for DIV2K and validation sets.
* **Deliverables**:
  * Validations checks and safe loaders `utils.py`
  * Downsamplings, blurs, noise, and compression degradations `degradation.py`
  * Aligned random crop augmentations `transforms.py`
  * Training and validation dataloaders `dataloader.py`
  * Sanity checking script `verify_dataset.py`

---

## Active & Upcoming Phases

### Phase 4A: Student Model Framework & Engineering Infrastructure 🚧
* **Description**: Construct sub-module class interfaces, PyTest test suites, model parameters summary logs, weights initializers, and reproducible experiment template setups.
* **Deliverables**:
  * Model utilities diagnostics `model_utils.py`
  * Weight initialization mappings `initialization.py`
  * Student modular layers skeletons (`feature_extractor.py`, `distillation_block.py`, `feature_fusion.py`, `reconstruction.py`, `student_model.py`)
  * SwinIR Wrapper interface `swinir_wrapper.py`
  * PyTest test suites `tests/models/`
  * Verification runner `verify_models.py`
  * Experiment settings YAML templates (`exp01_baseline.yaml`, `exp02_distillation.yaml`, `exp03_residual_scaling.yaml`)

### Phase 4B: Neural Network Model Implementation 📅
* **Description**: Implement real neural network layers inside student components and complete SwinIR model loading.
* **Upcoming Tasks**:
  * Conv2d and activation layers in student extractor.
  * Splitted and attention distillation conv layers in blocks.
  * Concatenation and 1x1 projection layers in fusion.
  * PyTorch PixelShuffle sub-pixel convolution in reconstruction.
  * Real weights loading from SwinIR checkpoint in teacher wrapper.
