# import torch
# from torch.profiler import profile, record_function, ProfilerActivity
# from main import ToyMeasurementControl

# # Start the profiler
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Profile CPU and, if available, CUDA operations.
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),  # Profiling schedule
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiling_output'),  # Save profiling data directory for TensorBoard
#     record_shapes=True,  # Record shapes of the tensors
#     profile_memory=True,  # Profile memory usage
#     with_stack=True  # Record call stack information
# ) as prof:
#         with record_function("model_inference"):
#             # Run the model
#             tmc = ToyMeasurementControl()
#             tmc.run()
#         prof.step()  # Save the profile data

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Example model and data
model = torch.nn.Linear(10, 5)
input = torch.randn(128, 10)

# Start the profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Profile CPU and, if available, CUDA operations.
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),  # Profiling schedule
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),  # Save profiling data to './logs' directory for TensorBoard
    record_shapes=True,  # Record shapes of the tensors
    profile_memory=True,  # Profile memory usage
    with_stack=True  # Record call stack information
) as prof:
    for _ in range(10):
        with record_function("model_inference"):
            model(input)
        prof.step()  # Move to the next profiling step
