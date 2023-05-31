optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src_fl.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src_fl.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "src_fl.models.sequence.SequenceModel",
    "lm": "src_fl.models.sequence.long_conv_lm.ConvLMHeadModel",
    "lm_simple": "src_fl.models.sequence.simple_lm.SimpleLMHeadModel",
    "vit_b_16": "src_fl.models.baselines.vit_all.vit_base_patch16_224",
}

layer = {
    "id": "src_fl.models.sequence.base.SequenceIdentity",
    "ff": "src_fl.models.sequence.ff.FF",
    "mha": "src_fl.models.sequence.mha.MultiheadAttention",
    "s4d": "src_fl.models.sequence.ssm.s4d.S4D",
    "s4_simple": "src_fl.models.sequence.ssm.s4_simple.SimpleS4Wrapper",
    "long-conv": "src_fl.models.sequence.long_conv.LongConv",
    "h3": "src_fl.models.sequence.h3.H3",
    "h3-conv": "src_fl.models.sequence.h3_conv.H3Conv",
    "hyena": "src_fl.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src_fl.models.sequence.hyena.HyenaFilter",
    "vit": "src_fl.models.sequence.mha.VitAttention",
}

callbacks = {
    "timer": "src_fl.callbacks.timer.Timer",
    "params": "src_fl.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src_fl.callbacks.progressive_resizing.ProgressiveResizing",
}
