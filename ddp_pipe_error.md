INFO 2020-12-08 17:02:36,382 ssl_dataset.py:  97: Using disk_folder labels from /scratch/imagenet_full_size/061417/train
INFO 2020-12-08 17:02:36,390 ssl_dataset.py:  97: Using disk_folder labels from /scratch/imagenet_full_size/061417/train
INFO 2020-12-08 17:02:36,391 ssl_dataset.py:  97: Using disk_folder labels from /scratch/imagenet_full_size/061417/train
INFO 2020-12-08 17:02:38,656 trainer_main.py: 205: Phase advanced. Rank: 1
INFO 2020-12-08 17:02:38,656 state_update_hooks.py:  90: Starting phase 0 [train]
INFO 2020-12-08 17:02:39,246 trainer_main.py: 205: Phase advanced. Rank: 0
INFO 2020-12-08 17:02:39,250 state_update_hooks.py:  90: Starting phase 0 [train]
Traceback (most recent call last):
  File "tools/run_distributed_engines.py", line 152, in <module>
    hydra_main(overrides=overrides)
  File "tools/run_distributed_engines.py", line 133, in hydra_main
    launch_distributed(
  File "tools/run_distributed_engines.py", line 72, in launch_distributed
    torch.multiprocessing.spawn(
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 200, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 158, in start_processes
    while not context.join():
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 119, in join
    raise Exception(msg)
Exception:

-- Process 1 terminated with the following error:
Traceback (most recent call last):
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
    fn(i, *args)
  File "/private/home/m1n/git/vissl/tools/run_distributed_engines.py", line 124, in _distributed_worker
    process_main(cfg, dist_run_id, local_rank=local_rank, node_id=node_id)
  File "/private/home/m1n/git/vissl/tools/run_distributed_engines.py", line 112, in process_main
    train_main(
  File "/private/home/m1n/git/vissl/vissl/engines/train.py", line 85, in train_main
    trainer.train()
  File "/private/home/m1n/git/vissl/vissl/trainer/trainer_main.py", line 115, in train
    task = train_step_fn(task)
  File "/private/home/m1n/git/vissl/vissl/trainer/train_steps/standard_train_step.py", line 177, in standard_train_step
    local_loss.backward()
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/tensor.py", line 198, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/autograd/__init__.py", line 98, in backward
    Variable._execution_engine.run_backward(
RuntimeError: grad.device() == bucket_view.device() INTERNAL ASSERT FAILED at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:238, please report a bug to PyTorch.  (mark_variable_ready_dense at /pytorch/torch/csrc/distrib
uted/c10d/reducer.cpp:238)
frame #0: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x46 (0x7efd578ac536 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::Reducer::mark_variable_ready_dense(c10d::Reducer::VariableIndex) + 0x4b0 (0x7efd9f1ec080 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #2: c10d::Reducer::mark_variable_ready(c10d::Reducer::VariableIndex) + 0x345 (0x7efd9f1ec9b5 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #3: c10d::Reducer::autograd_hook(c10d::Reducer::VariableIndex) + 0x1f2 (0x7efd9f1ed452 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #4: <unknown function> + 0x87197c (0x7efd9f1e797c in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #5: torch::autograd::Engine::evaluate_function(std::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&) + 0x60d (0x7efd9254391d in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so)
frame #6: torch::autograd::Engine::thread_main(std::shared_ptr<torch::autograd::GraphTask> const&, bool) + 0x3d2 (0x7efd925457e2 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so)
frame #7: torch::autograd::Engine::thread_init(int) + 0x39 (0x7efd9253de59 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so)
frame #8: torch::autograd::python::PythonEngine::thread_init(int) + 0x38 (0x7efd9ee81488 in /private/home/m1n/e/py38_vissl/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #9: <unknown function> + 0xbd66f (0x7efd9fd0b66f in /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
frame #10: <unknown function> + 0x76db (0x7efda1b1e6db in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #11: clone + 0x3f (0x7efda184788f in /lib/x86_64-linux-gnu/libc.so.6)


