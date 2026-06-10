# main.tf — provision N Yandex Cloud VMs on one private subnet + one public IP (bastion).
#
# Design: identical bastion pattern to the Selectel/OpenStack version.
#   - node-01 gets a public IP (NAT) — the only SSH entry point from the internet.
#   - nodes 02-N have private IPs only; reached via ProxyJump through node-01.
#   - A shared NAT gateway gives all private nodes outbound internet
#     (needed for apt-get, git clone, CIFAR10 download during bootstrap).
#
# After `terraform apply`, this writes ../../pool.yaml automatically.
#
# Quick start:
#   1. Install Yandex Cloud CLI:  https://cloud.yandex.ru/docs/cli/quickstart
#   2. Authenticate:              yc init
#   3. Get OAuth token:           export YC_TOKEN=$(yc iam create-token)
#   4. Copy tfvars:               cp terraform.tfvars.example terraform.tfvars
#   5. Fill in folder_id in terraform.tfvars
#   6. Render cloud-init:
#        python3 ../../../render_cloud_init.py \
#          --template cloud-init.yc.yaml.template \
#          --out cloud-init.rendered.yaml
#   7. terraform init && terraform apply

terraform {
  required_version = ">= 1.3"
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.120"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}

provider "yandex" {
  folder_id = var.folder_id
  zone      = var.zone
  # Auth: set YC_TOKEN env var (from `yc iam create-token`)
  # or set YC_SERVICE_ACCOUNT_KEY_FILE pointing to a JSON key file.
}

# ─── Network ─────────────────────────────────────────────────────────────────

resource "yandex_vpc_network" "net" {
  name = "${var.cluster_name}-net"
}

# NAT gateway — provides outbound internet for nodes without a public IP.
resource "yandex_vpc_gateway" "nat_gw" {
  name      = "${var.cluster_name}-nat-gw"
  folder_id = var.folder_id
  shared_egress_gateway {}
}

resource "yandex_vpc_route_table" "rt" {
  name       = "${var.cluster_name}-rt"
  network_id = yandex_vpc_network.net.id
  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat_gw.id
  }
}

resource "yandex_vpc_subnet" "subnet" {
  name           = "${var.cluster_name}-subnet"
  network_id     = yandex_vpc_network.net.id
  zone           = var.zone
  v4_cidr_blocks = [var.subnet_cidr]
  route_table_id = yandex_vpc_route_table.rt.id
}

# ─── Security group ───────────────────────────────────────────────────────────

resource "yandex_vpc_security_group" "sg" {
  name       = "${var.cluster_name}-sg"
  network_id = yandex_vpc_network.net.id

  # SSH from internet (key-only auth — bastion is the only node with a public IP).
  ingress {
    protocol       = "TCP"
    port           = 22
    v4_cidr_blocks = [var.ssh_allowed_cidr]
    description    = "SSH from internet"
  }

  # gRPC + HTTP config server — only from within the private subnet.
  ingress {
    protocol       = "TCP"
    from_port      = var.grpc_port
    to_port        = var.grpc_port + 1
    v4_cidr_blocks = [var.subnet_cidr]
    description    = "gRPC + config server intra-cluster"
  }

  # All traffic between members of this security group (covers bastion→follower SSH
  # and all inter-node gRPC).
  ingress {
    protocol          = "ANY"
    predefined_target = "self_security_group"
    description       = "Intra-cluster all"
  }

  # Unrestricted egress (apt, git clone, CIFAR-10 download, peer weight pushes).
  egress {
    protocol       = "ANY"
    v4_cidr_blocks = ["0.0.0.0/0"]
    description    = "All outbound"
  }
}

# ─── Image ────────────────────────────────────────────────────────────────────

data "yandex_compute_image" "ubuntu" {
  family = var.image_family
}

# ─── Nodes ────────────────────────────────────────────────────────────────────

resource "yandex_compute_instance" "node" {
  count       = var.server_count
  name        = format("%s-%02d", var.cluster_name, count.index + 1)
  hostname    = format("%s-%02d", var.cluster_name, count.index + 1)
  platform_id = var.platform_id
  zone        = var.zone
  folder_id   = var.folder_id

  resources {
    cores         = var.node_cpu_cores
    memory        = var.node_memory_gb
    core_fraction = var.core_fraction
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.volume_size_gb
      type     = var.disk_type
    }
  }

  network_interface {
    subnet_id          = yandex_vpc_subnet.subnet.id
    security_group_ids = [yandex_vpc_security_group.sg.id]
    # Only node-01 (index 0) gets a public IP — it is the bastion/jump host.
    # All other nodes are private; they are reached via ProxyJump.
    nat = count.index == 0
  }

  metadata = {
    # user-data drives cloud-init: creates the decentr user, installs Python,
    # clones the project repo, and downloads CIFAR-10.
    user-data = file(var.cloud_init_path)
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  lifecycle {
    ignore_changes = [boot_disk[0].initialize_params[0].image_id]
  }
}

# ─── Auto-generate ../../pool.yaml ───────────────────────────────────────────

locals {
  bastion_public_ip = yandex_compute_instance.node[0].network_interface[0].nat_ip_address
  nodes = [
    for i in range(var.server_count) : {
      id         = format("decentr-%02d", i + 1)
      private_ip = yandex_compute_instance.node[i].network_interface[0].ip_address
      # node-01 is reached via its public (NAT) IP.
      # All other nodes are accessed via ProxyJump through node-01.
      ssh_host   = i == 0 ? yandex_compute_instance.node[0].network_interface[0].nat_ip_address : yandex_compute_instance.node[i].network_interface[0].ip_address
      ssh_jump   = i == 0 ? "" : yandex_compute_instance.node[0].network_interface[0].nat_ip_address
    }
  ]
}

resource "local_file" "pool" {
  filename = "${path.module}/../../pool.yaml"
  content = templatefile("${path.module}/templates/pool.yaml.tftpl", {
    pool_id      = var.cluster_name
    ssh_key_path = var.ssh_private_key_path
    cpu_cores    = var.node_cpu_cores
    nodes        = local.nodes
  })
  depends_on = [yandex_compute_instance.node]
}
