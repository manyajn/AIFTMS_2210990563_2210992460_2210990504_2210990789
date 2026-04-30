# AIFT-MS Framework
AI-Integrated Finance Transformation for Mid-Sized Companies

## Overview
A research-backed conceptual and technical framework for enabling AI-driven finance transformation in mid-sized enterprises.

This repository contains implementation patterns for:
- Revenue Forecasting using LightGBM
- Fraud Detection using Isolation Forest
- LLM-Powered Financial Narrative Generation
- Terraform Infrastructure as Code
- Kubernetes Deployment Manifests

## Architecture
The framework follows a 5-tier modular architecture:

1. Data Ingestion & Governance
2. Integration & Real-Time Streaming
3. AI/ML Engine
4. Automation & Orchestration
5. Intelligence UX
-  Visual Diagram for same:
  
 ERP/CRM/Bank APIs
      ↓
Tier 1: Data Ingestion
      ↓
Tier 2: Kafka / Streaming
      ↓
Tier 3: AI/ML Models
      ↓
Tier 4: Automation Bots
      ↓
Tier 5: CFO Dashboard

## Tech Stack
- Python
- LightGBM
- Kafka
- Terraform
- Kubernetes
- Snowflake
- Docker
- MLflow
- Feast Feature Store

## Repository Structure
```bash
finance_forecast/model/lgbm_revenue.py
fraud_detection/consumer/kafka_scorer.py
narrative/generator.py
infrastructure/modules/snowflake/main.tf
k8s/production/revenue-forecast-deployment.yaml
