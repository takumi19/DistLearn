# outputs.tf — useful values after `terraform apply`.

output "floating_ips" {
  description = "Public IPs (use for SSH). Index 0 = node-01."
  value       = openstack_networking_floatingip_v2.fip[*].address
}

output "private_ips" {
  description = "Private subnet IPs (used for inter-node gRPC)."
  value       = openstack_networking_port_v2.port[*].all_fixed_ips[0]
}

output "node_names" {
  description = "OpenStack instance names."
  value       = openstack_compute_instance_v2.node[*].name
}

output "pool_yaml_path" {
  description = "Where pool.yaml was written."
  value       = abspath("${path.module}/../../pool.yaml")
}

output "ssh_hint" {
  description = "Quick SSH command for node-01."
  value       = "ssh -i ${var.ssh_private_key_path} decentr@${openstack_networking_floatingip_v2.fip[0].address}"
}
