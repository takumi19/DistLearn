# outputs.tf — useful values after `terraform apply`.

output "bastion_public_ip" {
  description = "Public IP of node-01 (bastion). SSH entry point."
  value       = yandex_compute_instance.node[0].network_interface[0].nat_ip_address
}

output "private_ips" {
  description = "Private subnet IPs (used for inter-node gRPC)."
  value       = [for n in yandex_compute_instance.node : n.network_interface[0].ip_address]
}

output "node_names" {
  description = "Instance names."
  value       = [for n in yandex_compute_instance.node : n.name]
}

output "pool_yaml_path" {
  description = "Where pool.yaml was written."
  value       = abspath("${path.module}/../../pool.yaml")
}

output "ssh_hint" {
  description = "Quick SSH command for the bastion (node-01)."
  value       = "ssh -i ${var.ssh_private_key_path} decentr@${yandex_compute_instance.node[0].network_interface[0].nat_ip_address}"
}
