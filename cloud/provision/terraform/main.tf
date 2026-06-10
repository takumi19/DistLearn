# main.tf — provision N Selectel cloud servers on one private subnet + floating IPs.
#
# Design: all nodes live in ONE Selectel project on ONE private subnet, so they
# reach each other directly by private IP (no Tailscale needed for inter-node
# gRPC). Each node also gets a floating (public) IP so your laptop's orchestrator
# can SSH in.
#
# After `terraform apply`, this writes ../../pool.yaml automatically.
#
# Prereqs (one-time, in https://my.selectel.ru/):
#   - create a project (note its UUID -> project_id)
#   - create a service user with the `member` role on that project
#     (Identity & Access Management -> User management -> Service users)
#   - render cloud-init:  python3 ../../render_cloud_init.py \
#         --template cloud-init.subnet.yaml.template \
#         --out cloud-init.rendered.yaml

terraform {
  required_version = ">= 1.3"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 2.1"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}

provider "openstack" {
  auth_url    = var.auth_url
  domain_name = var.domain_name # Selectel account ID
  tenant_id   = var.project_id  # project UUID
  user_name   = var.username    # service user
  password    = var.password
  region      = var.region
}

# --- Networking -------------------------------------------------------------

resource "openstack_networking_network_v2" "net" {
  name           = "${var.cluster_name}-net"
  admin_state_up = "true"
}

resource "openstack_networking_subnet_v2" "subnet" {
  name            = "${var.cluster_name}-subnet"
  network_id      = openstack_networking_network_v2.net.id
  cidr            = var.subnet_cidr
  ip_version      = 4
  dns_nameservers = ["188.93.16.19", "188.93.17.19"] # Selectel resolvers
}

data "openstack_networking_network_v2" "external" {
  external = true
}

resource "openstack_networking_router_v2" "router" {
  name                = "${var.cluster_name}-router"
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "router_iface" {
  router_id = openstack_networking_router_v2.router.id
  subnet_id = openstack_networking_subnet_v2.subnet.id
}

# --- Security group ---------------------------------------------------------

resource "openstack_networking_secgroup_v2" "sg" {
  name        = "${var.cluster_name}-sg"
  description = "decentr cloud nodes: ssh + intra-cluster gRPC"
}

# SSH from anywhere (key-only auth on the box).
resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = var.ssh_allowed_cidr
  security_group_id = openstack_networking_secgroup_v2.sg.id
}

# gRPC + HTTP config server, only within the private subnet.
resource "openstack_networking_secgroup_rule_v2" "grpc" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = var.grpc_port
  port_range_max    = var.grpc_port + 1 # config server = grpc_port + 1
  remote_ip_prefix  = var.subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.sg.id
}

# Allow all traffic between members of this security group.
resource "openstack_networking_secgroup_rule_v2" "intra" {
  direction         = "ingress"
  ethertype         = "IPv4"
  remote_group_id   = openstack_networking_secgroup_v2.sg.id
  security_group_id = openstack_networking_secgroup_v2.sg.id
}

# --- Keypair + image --------------------------------------------------------

resource "openstack_compute_keypair_v2" "key" {
  name       = "${var.cluster_name}-key"
  public_key = file(var.ssh_public_key_path)
}

data "openstack_images_image_v2" "ubuntu" {
  name        = var.image_name
  most_recent = true
  visibility  = "public"
}

# --- Per-node resources (count = var.server_count) --------------------------

resource "openstack_networking_port_v2" "port" {
  count          = var.server_count
  name           = format("%s-%02d-port", var.cluster_name, count.index + 1)
  network_id     = openstack_networking_network_v2.net.id
  admin_state_up = "true"
  security_group_ids = [openstack_networking_secgroup_v2.sg.id]
  fixed_ip {
    subnet_id = openstack_networking_subnet_v2.subnet.id
  }
}

resource "openstack_blockstorage_volume_v3" "boot" {
  count                = var.server_count
  name                 = format("%s-%02d-boot", var.cluster_name, count.index + 1)
  size                 = var.volume_size_gb
  image_id             = data.openstack_images_image_v2.ubuntu.id
  volume_type          = var.volume_type
  availability_zone    = var.availability_zone
  enable_online_resize = true
  lifecycle {
    ignore_changes = [image_id]
  }
}

resource "openstack_compute_instance_v2" "node" {
  count             = var.server_count
  name              = format("%s-%02d", var.cluster_name, count.index + 1)
  flavor_id         = var.flavor_id
  key_pair          = openstack_compute_keypair_v2.key.name
  availability_zone = var.availability_zone
  user_data         = file(var.cloud_init_path)

  network {
    port = openstack_networking_port_v2.port[count.index].id
  }

  block_device {
    uuid             = openstack_blockstorage_volume_v3.boot[count.index].id
    source_type      = "volume"
    destination_type = "volume"
    boot_index       = 0
  }

  vendor_options {
    ignore_resize_confirmation = true
  }

  lifecycle {
    ignore_changes = [image_id]
  }
}

# --- Floating IP: only node-01 (index 0) is the bastion/jump host -----------
# Default Selectel quota is ~12 floating IPs for a new project.
# Only the bastion needs a public IP; all other nodes are accessed via
# SSH ProxyJump through the bastion over the private subnet.

resource "openstack_networking_floatingip_v2" "fip" {
  count = 1
  pool  = var.external_network_name
}

resource "openstack_networking_floatingip_associate_v2" "fip_assoc" {
  count       = 1
  floating_ip = openstack_networking_floatingip_v2.fip[0].address
  port_id     = openstack_networking_port_v2.port[0].id
}

# --- Auto-generate ../../pool.yaml ------------------------------------------

locals {
  bastion_ip = openstack_networking_floatingip_v2.fip[0].address
  nodes = [
    for i in range(var.server_count) : {
      id          = format("decentr-%02d", i + 1)
      private_ip  = openstack_networking_port_v2.port[i].all_fixed_ips[0]
      # node-01 is reached directly via its floating IP;
      # all other nodes jump through node-01.
      ssh_host    = i == 0 ? openstack_networking_floatingip_v2.fip[0].address : openstack_networking_port_v2.port[i].all_fixed_ips[0]
      ssh_jump    = i == 0 ? "" : openstack_networking_floatingip_v2.fip[0].address
    }
  ]
}

resource "local_file" "pool" {
  filename = "${path.module}/../../pool.yaml"
  content = templatefile("${path.module}/templates/pool.yaml.tftpl", {
    pool_id       = var.cluster_name
    ssh_key_path  = var.ssh_private_key_path
    cpu_cores     = var.node_cpu_cores
    nodes         = local.nodes
  })
  # Floating IP association must exist before we trust the addresses.
  depends_on = [openstack_networking_floatingip_associate_v2.fip_assoc]
}
