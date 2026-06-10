# variables.tf — all knobs for the Selectel provisioning.
# Copy terraform.tfvars.example -> terraform.tfvars and fill in.

# --- Selectel auth (service user on a project) ------------------------------

variable "auth_url" {
  type        = string
  default     = "https://cloud.api.selcloud.ru/identity/v3"
  description = "Keystone identity endpoint."
}

variable "domain_name" {
  type        = string
  description = "Selectel account ID (top-right in my.selectel.ru)."
}

variable "username" {
  type        = string
  description = "Service user name (IAM -> User management -> Service users)."
}

variable "password" {
  type        = string
  sensitive   = true
  description = "Service user password."
}

variable "project_id" {
  type        = string
  description = "Project UUID (the OpenStack tenant_id)."
}

variable "region" {
  type        = string
  default     = "ru-9"
  description = "Selectel pool/region, e.g. ru-1, ru-3, ru-7, ru-8, ru-9."
}

variable "availability_zone" {
  type        = string
  default     = "ru-9a"
  description = "AZ within the region. Must match region (ru-9 -> ru-9a)."
}

# --- Sizing -----------------------------------------------------------------

variable "server_count" {
  type        = number
  default     = 20
  description = "How many nodes to provision."
}

variable "flavor_id" {
  type        = string
  description = "Flavor id. Find via panel or `python3 ../selectel.py list-flavors`. ~4 vCPU / 8 GB recommended."
}

variable "image_name" {
  type        = string
  default     = "Ubuntu 24.04 LTS 64-bit"
  description = "Base image. Must ship Python 3.12 (Ubuntu 24.04)."
}

variable "volume_size_gb" {
  type        = number
  default     = 40
  description = "Boot volume size. torch + CIFAR10 + checkpoints need ~40 GB."
}

variable "volume_type" {
  type        = string
  default     = "fast.ru-9a"
  description = "Selectel volume type, zone-specific (e.g. fast.ru-9a, basic.ru-9a)."
}

variable "node_cpu_cores" {
  type        = number
  default     = 4
  description = "Cores per node — written into pool.yaml (cosmetic; affects scheduler hints)."
}

# --- Networking -------------------------------------------------------------

variable "cluster_name" {
  type        = string
  default     = "decentr"
  description = "Prefix for all created resources and pool_id."
}

variable "subnet_cidr" {
  type        = string
  default     = "192.168.199.0/24"
  description = "Private subnet for inter-node gRPC. /24 fits 250 nodes."
}

variable "external_network_name" {
  type        = string
  default     = "external-network"
  description = "Floating IP pool name. Usually 'external-network' on Selectel."
}

variable "ssh_allowed_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "Who may SSH in. Tighten to your IP/32 for better security."
}

variable "grpc_port" {
  type        = number
  default     = 50051
  description = "gRPC port. Config server runs on grpc_port + 1."
}

# --- SSH + cloud-init -------------------------------------------------------

variable "ssh_public_key_path" {
  type        = string
  default     = "~/.ssh/decentr_id_ed25519.pub"
  description = "Public key uploaded to nodes (orchestrator SSH)."
}

variable "ssh_private_key_path" {
  type        = string
  default     = "~/.ssh/decentr_id_ed25519"
  description = "Private key path — written into pool.yaml for the orchestrator."
}

variable "cloud_init_path" {
  type        = string
  default     = "cloud-init.rendered.yaml"
  description = "Rendered cloud-init (relative to the terraform/ dir)."
}
