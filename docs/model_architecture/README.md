# Super-Resolution Model Architecture & Framework Documentation

This directory details the engineering framework and specifications for the Student model, Teacher wrapper, and model utilities.

## Folder Organization

All model files are organized modularly under `training/models/`:

```
models/
├── student/
│   ├── __init__.py
│   ├── feature_extractor.py    # Shallow extraction layers
│   ├── distillation_block.py   # Split distillation block blocks (RFDB / IMDB)
│   ├── feature_fusion.py       # Global block-wise feature concatenation
│   ├── reconstruction.py       # PixelShuffle upsampling back-end
│   └── student_model.py        # Top-level module assembly coordinating flow
├── teacher/
│   ├── __init__.py
│   └── swinir_wrapper.py       # Pretrained SwinIR teacher API wrapper
└── utils/
    ├── __init__.py
    ├── initialization.py       # Kaiming/Xavier weights initialization
    └── model_utils.py          # Sizing, parameters counting, device audits
```

---

## Architectural Flow Pipeline

The Student model is designed to resemble the **Residual Feature Distillation Network (RFDN)** structure:

```
Input LR Image ──> Shallow Conv Extractor (feature_extractor)
                         │
                         ▼
             Distillation Blocks list (distillation_block) [seq 1..N]
                         │
                         ├─► Block 1 feature map ──────┐
                         ├─► Block 2 feature map ──────┼─► Feature Fusion (feature_fusion)
                         └─► Block N feature map ──────┘
                                                             │
                                                             ▼
                                                Fused global representations
                                                             │
                                           (Skip Add)        ▼
                                       [Fused features + Shallow features]
                                                             │
                                                             ▼
                                             Reconstruction (reconstruction)
                                                             │
                                                             ▼
                                                       Upscaled HR image
```

---

## Coding Conventions & Future Implementations

1. **Configurations Integration**: No model layers or parameters should be hardcoded. Everything should be resolved from `training.utils.config` (`ExperimentConfig`).
2. **Type Hints**: All functions and arguments must specify explicit input/output type mappings.
3. **Weight Initializations**: In Phase 4B, initialize weights using `custom_initialize(model, method='kaiming')` after model creation.
4. **Testing Philosophy**: All components must be unit-tested inside `tests/models/` for shape compatibility before merging implementation edits.
