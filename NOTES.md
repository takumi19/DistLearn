
## Datasets
1. ImageNet (1.2 million images; 1000 classes; ~140GB compressed, ~300GB uncompressed)
2. CIFAR-10 (60k images; <1GB size)
3. MNIST (70k images; <1GB size)
4. LFW (fits in RAM too)

## Neural Networks
1. Swin
2. ConvNeXt
3. YOLO
4. ResNet

## Architectures and approaches

### Data parallelism vs model parallelism

- *Data parallelism*. 
-   *Synchronous*: Data is split into several batches and each device processes one of them. Each device holds an identical copy of the model. Before the first iteration they are all initialized with the same set of random weights, which are likely generated and sent out by the leader (rank=0 machine in MPI terminology), or generated using the same seed on each machine. After each iteration the resulting gradients from all devices are averaged, and the average gradient is sent back to all the machines and weights are updated.
-   *Asynchronous*: Workers run their computation independently and update the weights in a distributed parameter storage. They do not wait for other workers, instead they take the parameters that are in the storage. This means that some workers will use stale parameters, which lowers the statistical learning efficiency.
- *Model parallelism*. The model's layers are split across several devices. For example, GPU0 has conv1, GPU1 has conv2 and conv3, GPU2 has fc1 and fc2. Good for large models, since the entire model does not have to sit in memory of a single computer. If the GPU has 16GB of memory and each parameter is 4 bytes, then this means there ought to be more than a trillion parameters (?).
- *Pipeline parallelism*: combines model and data parallelism. Each minibatch is divided into several *microbatches*. Lets say we have three devices and we divide each minibatch in three microbatches. 
1. Device 1 -> microbatch 1.
2. Device 1 -> microbatch 2. Device 2 -> microbatch 1.
3. Device 1 -> microbatch 3. Device 2 -> microbatch 2. Device 3 -> microbatch 1.
4.                           Device 2 -> microbatch 3. Device 3 -> microbatch 2.
5.                                                     Device 3 -> microbatch 3.
... backprop


When to use:
1. Model parallelism:
1.1. For large models (e.g., BERT, GPT) with billions of parameters that exceed a single GPU’s memory. Has more communication overhead.
2. Data parallelism:
2.1. For large datasets

## Q
1. Not entirely clear how the parameters are distributed across devices in Pipeline parallelism.

## Frameworks

Support DDP:
- [PyTorch DistributedDataParallel, DDP](): simplest one, 
- [PyTorch lightning]() supports various strategies;
- [Horovod](https://www.uber.com/blog/horovod/?uclick_id=3f23ceba-b1f1-4c15-9c0f-0f050e004128) - Primarily focuses on Data parallelism using efficient ring_allreduce and TensorFusion.
- [TensorFlow Distributed Strategy]() - supports Asynchronous data parallelism with a parameter server strategy.

Support model parallelism:
- Pytorch and TF native support using multiple GPUs on one computer.
- MegaTron-LM supports multi-node parallelism, built on Pytorch
- DeepSpeed supports multi-node parallelism, built on Pytorch

Pipeline parallelism:
- torch.distributed.pipeline.sync.Pipe: GPipe style pipelining
- DataSpeed, GPipe style too
- MegaTron-LM

Other:
- DeepSpeed, MegaTron, support pretty much everything
- Horovod (Elastic), TorchElastic support dynamic node management and provide higher availability and fault tolerance
- [Ray](https://www.oreilly.com/library/view/learning-ray/9781098117214/): easy integration with Pytorch; supports asynchronous DDP, model parallelism, etc.
- Spark + MapReduce
"While the MapReduce model was successful for deep learning at first, its generality hindered
DNN-specific optimizations. Therefore, current implementations make use of high-performance
communication interfaces (e.g., MPI) to implement fine-grained parallelism features, such as
reducing latencies via asynchronous execution and pipelining"
