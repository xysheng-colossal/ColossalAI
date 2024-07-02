import argparse
import resource
import time
from contextlib import nullcontext

import torch
from calflops import calculate_flops
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel
from performance_evaluator import PerformanceEvaluator, get_profile_context
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer import PipelineGradientCheckpointConfig

# ==============================
# Constants
# ==============================

MODEL_CONFIGS = {
    "900m": Qwen2Config(
        hidden_size=512,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=32,
        max_position_embeddings=4096,
    ),
    "7b": Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=131072,
    ),
    "72b": Qwen2Config(
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
    ),
}

calflops_cache = {"7b": 64665858080768}


# Calculate the flops of one batch of models
def get_calflops(model_config, max_length):
    print(
        "                 The calflops tool is being used to calculate flops. \n \
            Please wait for a few minutes. If there is already flops data, \n \
            you can add it to the calflops_cache and the flops will not be recalculated."
    )
    batch_size = 1
    vocab_size = model_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, max_length), device=get_accelerator().get_current_device())
    attention_mask = torch.ones_like(input_ids)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }
    model = AutoModelForCausalLM.from_config(
        model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    flops, macs, params = calculate_flops(
        model=model, kwargs=model_inputs, print_results=False, print_detailed=False, output_as_string=False
    )
    print("Qwen2 FLOPs:%d   MACs:%d   Params:%d \n" % (flops, macs, params))
    return flops


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "fsdp", "fsdp_cpu", "3d", "3d_cpu"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=2, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument("-m", "--memory_limit", type=int, help="Gemini memory limit in mb")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use xformers")
    parser.add_argument("--shard_param_frac", type=float, default=1.0, help="Shard param fraction. Only for gemini")
    parser.add_argument("--offload_optim_frac", type=float, default=0.0, help="Offload optim fraction. Only for gemini")
    parser.add_argument("--offload_param_frac", type=float, default=0.0, help="Offload param fraction. Only for gemini")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--sp", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--extra_dp", type=int, default=1, help="Extra data parallel size, used for Gemini")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size of pipeline parallel")
    parser.add_argument("--zero", type=int, default=0, help="Zero Stage when hybrid plugin is enabled")
    parser.add_argument("--custom-ckpt", action="store_true", help="Customize checkpoint", default=False)
    parser.add_argument("--profile", action="store_true", help="Enable profiling", default=False)
    parser.add_argument(
        "--disable-async-reduce", action="store_true", help="Disable the asynchronous reduce operation", default=False
    )
    parser.add_argument("--prefetch_num", type=int, default=0, help="chunk prefetch max number")
    parser.add_argument("--calflops", action="store_true", help="Use calflops to calculating flops")
    args = parser.parse_args()

    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    def empty_init():
        pass

    # ckpt config for LLaMA3-70B on 64 H100 GPUs
    hybrid_kwargs = (
        {
            "gradient_checkpoint_config": PipelineGradientCheckpointConfig(
                num_ckpt_layers_per_stage=[19, 19, 19, 13],
            ),
            "num_layers_per_stage": [19, 20, 20, 21],
        }
        if args.custom_ckpt
        else {}
    )

    # ==============================
    # Initialize Booster
    # ==============================
    use_empty_init = True
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision="bf16",
            shard_param_frac=args.shard_param_frac,
            offload_optim_frac=args.offload_optim_frac,
            offload_param_frac=args.offload_param_frac,
            tp_size=args.tp,
            extra_dp_size=args.extra_dp,
            enable_fused_normalization=torch.cuda.is_available(),
            enable_flash_attention=args.xformers,
            max_prefetch=args.prefetch_num,
            enable_async_reduce=not args.disable_async_reduce,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            placement_policy="auto",
            precision="bf16",
            warmup_non_model_data_ratio=args.warmup_ratio,
            tp_size=args.tp,
            extra_dp_size=args.extra_dp,
            enable_fused_normalization=torch.cuda.is_available(),
            max_prefetch=args.prefetch_num,
            enable_async_reduce=not args.disable_async_reduce,
            enable_flash_attention=args.xformers,
        )
    elif args.plugin == "fsdp":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                param_init_fn=empty_init(),
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            )
    elif args.plugin == "fsdp_cpu":
        if use_empty_init:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                cpu_offload=CPUOffload(offload_params=True),
                param_init_fn=empty_init(),
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                ),
                cpu_offload=CPUOffload(offload_params=True),
            )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            zero_stage=args.zero,
            sp_size=args.sp,
            enable_sequence_parallelism=args.sp > 1,
            enable_fused_normalization=torch.cuda.is_available(),
            enable_flash_attention=args.xformers,
            microbatch_size=args.mbs,
            precision="bf16",
            dp_outside=False,
            **hybrid_kwargs,
        )
    elif args.plugin == "3d_cpu":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            zero_stage=args.zero,
            cpu_offload=True,
            enable_fused_normalization=torch.cuda.is_available(),
            enable_flash_attention=args.xformers,
            microbatch_size=args.mbs,
            initial_scale=2**8,
            precision="bf16",
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    dp_size = getattr(plugin, "dp_size", coordinator.world_size)

    if args.config in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.config]
    else:
        config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)
    dataset = RandomDataset(
        num_samples=args.batch_size * args.num_steps * dp_size, max_length=args.max_length, vocab_size=config.vocab_size
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if isinstance(plugin, (GeminiPlugin, HybridParallelPlugin))
        else nullcontext()
    )

    init_kwargs = {}
    if config.model_type == "chatglm":
        init_kwargs["empty_init"] = False

    with init_ctx:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            **init_kwargs,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if config.model_type == "chatglm":
            model.transformer.encoder.gradient_checkpointing = True

    model_numel = get_model_numel(model)
    model_calflops = (
        calflops_cache[args.config]
        if args.config in calflops_cache and not args.calflops
        else get_calflops(MODEL_CONFIGS[args.config], args.max_length)
    )
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        model_calflops,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.vocab_size,
        args.grad_checkpoint,
        args.ignore_steps,
        dp_world_size=dp_size,
    )

    optimizer = HybridAdam(model.parameters())
    torch.set_default_dtype(torch.bfloat16)
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)
    torch.set_default_dtype(torch.float)
    coordinator.print_on_master(
        f"Booster init max CUDA memory: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB"
    )
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    with get_profile_context(
        args.profile,
        1,
        len(dataloader) - 1,
        save_dir=f"profile/{time.strftime('%H:%M', time.localtime())}-{args.plugin}-qwen2-{args.config}",
    ) as prof:
        if isinstance(plugin, HybridParallelPlugin) and args.pp > 1:
            data_iter = iter(dataloader)
            for step in tqdm(range(len(dataloader)), desc="Step", disable=not coordinator.is_master()):
                performance_evaluator.on_step_start(step)
                booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=False,
                )
                optimizer.step()
                optimizer.zero_grad()
                performance_evaluator.on_step_end(input_ids=torch.empty(args.batch_size, args.max_length))
                prof.step()
        else:
            for step, batch in enumerate(tqdm(dataloader, desc="Step", disable=not coordinator.is_master())):
                performance_evaluator.on_step_start(step)
                outputs = model(**batch)
                loss = outputs[0]
                booster.backward(loss, optimizer)
                optimizer.step()
                optimizer.zero_grad()
                performance_evaluator.on_step_end(**batch)
                prof.step()

    performance_evaluator.on_fit_end()
    coordinator.print_on_master(f"Max CUDA memory usage: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()