# variables.tf — all knobs for Yandex Cloud provisioning.
# Copy terraform.tfvars.example → terraform.tfvars and fill in folder_id.

# ─── Yandex Cloud identity ────────────────────────────────────────────────────

variable "folder_id" {
  type        = string
  description = "Yandex Cloud folder ID. Find in the console URL or `yc resource-manager folder list`."
}

variable "zone" {
  type        = string
  default     = "ru-central1-a"
  description = "Availability zone. Options: ru-central1-a/b/c/d."
}

# ─── Sizing ───────────────────────────────────────────────────────────────────

variable "server_count" {
  type        = number
  default     = 20
  description = "How many nodes to provision."
}

variable "platform_id" {
  type        = string
  default     = "standard-v3"
  description = "CPU platform. standard-v3 = Intel Ice Lake (recommended). standard-v2 = Cascade Lake."
}

variable "node_cpu_cores" {
  type        = number
  default     = 2
  description = "vCPU cores per node. Also written into pool.yaml for the scheduler."
}

variable "node_memory_gb" {
  type        = number
  default     = 4
  description = "RAM (GB) per node. 4 GB is sufficient for CIFAR-10/ResNet-18 on CPU."
}

variable "core_fraction" {
  type        = number
  default     = 100
  description = "Guaranteed CPU fraction (%). 100 = dedicated. 20/50 = burstable (cheaper, not for training)."
}

variable "preemptible" {
  type        = bool
  default     = false
  description = "Use preemptible instances (~30% cheaper, but Yandex can stop them after 24h)."
}

variable "image_family" {
  type        = string
  default     = "ubuntu-2404-lts"
  description = "Yandex compute image family. ubuntu-2404-lts ships Python 3.12 natively."
}

variable "volume_size_gb" {
  type        = number
  default     = 40
  description = "Boot disk size (GB). torch + CIFAR-10 + checkpoints need ~40 GB."
}

variable "disk_type" {
  type        = string
  default     = "network-ssd"
  description = "Disk type. network-ssd recommended for ML I/O. network-hdd is cheaper."
}

# ─── Networking ───────────────────────────────────────────────────────────────

variable "cluster_name" {
  type        = string
  default     = "decentr"
  description = "Prefix for all created resources and pool_id in pool.yaml."
}

variable "subnet_cidr" {
  type        = string
  default     = "192.168.199.0/24"
  description = "Private subnet CIDR. /24 fits 250 nodes."
}

variable "grpc_port" {
  type        = number
  default     = 50051
  description = "gRPC port. Config server runs on grpc_port + 1 (50052)."
}

variable "ssh_allowed_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "Who may SSH into the bastion. Tighten to your IP/32 for security."
}

# ─── SSH + cloud-init ─────────────────────────────────────────────────────────

variable "ssh_public_key_path" {
  type        = string
  default     = "~/.ssh/decentr_id_ed25519.pub"
  description = "Public key embedded in cloud-init (nodes accept only this key)."
}

variable "ssh_private_key_path" {
  type        = string
  default     = "~/.ssh/decentr_id_ed25519"
  description = "Private key path — written into pool.yaml for the orchestrator."
}

variable "cloud_init_path" {
  type        = string
  default     = "cloud-init.rendered.yaml"
  description = "Rendered cloud-init file (relative to terraform-yc/ dir). Render with render_cloud_init.py first."
}
