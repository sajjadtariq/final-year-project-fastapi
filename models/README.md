# Local model files

Place or reference the two model checkpoints outside source control:

```text
models/
├── efficientnet-b3/
│   └── checkpoints/
│       └── best.pt
└── yolo11m.pt
```

The EfficientNet checkpoint is selected with
`SKINHEALTH_EFFICIENTNET_CHECKPOINT=./models/efficientnet-b3/checkpoints/best.pt`.
Either model may instead remain elsewhere and be selected through its corresponding
environment variable. All `.pt` files are ignored by the parent repository and must
be transferred through the project's approved private model store.

Do not rename or substitute a checkpoint without also updating its immutable model
version and verifying its class mapping.

<!-- File summary: Explains where local checkpoints come from and why model binaries
are deliberately excluded from the copyable source-code handoff. -->
