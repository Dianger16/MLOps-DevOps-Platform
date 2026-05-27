# terraform/main.tf
# Provisions minimal EKS for MLOps model serving
# Cost: ~$0.10/hr control plane + ~$0.021/hr per t3.small node
# DESTROY WHEN DONE

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = { Project = "mlops-platform", ManagedBy = "Terraform" }
  }
}

variable "aws_region"   { default = "us-east-1" }
variable "cluster_name" { default = "mlops-platform" }

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name            = "${var.cluster_name}-vpc"
  cidr            = "10.0.0.0/16"
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags  = { "kubernetes.io/role/elb" = "1" }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = "1" }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.31"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    main = {
      instance_types = ["t3.small"]   # 2GB RAM — enough for serving
      min_size       = 1
      max_size       = 3
      desired_size   = 2
    }
  }
}

# ── ECR repository for model serving image ────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "sentiment-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

output "cluster_name"      { value = module.eks.cluster_name }
output "ecr_repository_url" { value = aws_ecr_repository.api.repository_url }
output "configure_kubectl" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${var.cluster_name}"
}
output "cost_warning" {
  value = "~$0.14/hr (~$3.36/day). Run terraform destroy when done!"
}
