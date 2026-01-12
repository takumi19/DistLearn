from datetime import timedelta
import os
import torch.distributed as dist

def main():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        timeout=timedelta(seconds=50),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Hello from rank {rank} out of {world_size}!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
