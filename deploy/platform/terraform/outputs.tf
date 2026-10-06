output "public_ip" {
  value = oci_core_instance.platform.public_ip
}

output "instance_id" {
  value = oci_core_instance.platform.id
}

output "grafana_url" {
  value = "https://grafana.${oci_core_instance.platform.public_ip}.sslip.io"
}
