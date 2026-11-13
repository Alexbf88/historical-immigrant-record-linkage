terraform {
  required_version = ">= 1.5.0"

  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.0"
    }
  }
}

variable "github_token" {
  description = "GitHub Personal Access Token (or pass via GITHUB_TOKEN env var)"
  type        = string
  sensitive   = true
  default     = null
}

variable "repository_name" {
  description = "Name of the GitHub repository to provision"
  type        = string
  default     = "historical-immigrant-record-linkage"
}

provider "github" {
  token = var.github_token
}

resource "github_repository" "portfolio_repo" {
  name        = var.repository_name
  description = "End-to-end NLP & Random Forest Entity Resolution pipeline for historical immigrant arrival records (Playwright, Scikit-Learn, FastAPI, Kubernetes & Terraform)"
  visibility  = "public"

  has_issues      = true
  has_wiki        = false
  has_projects    = false
  has_discussions = false

  topics = [
    "data-science",
    "machine-learning",
    "nlp",
    "record-linkage",
    "entity-resolution",
    "playwright",
    "fastapi",
    "kubernetes",
    "terraform",
    "sre",
    "finops"
  ]
}

output "repository_html_url" {
  description = "Public URL of the provisioned GitHub repository"
  value       = github_repository.portfolio_repo.html_url
}

output "git_remote_https" {
  description = "HTTPS Git remote URL"
  value       = github_repository.portfolio_repo.http_clone_url
}

output "git_remote_ssh" {
  description = "SSH Git remote URL"
  value       = github_repository.portfolio_repo.ssh_clone_url
}
