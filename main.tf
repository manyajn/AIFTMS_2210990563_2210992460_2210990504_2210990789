apiVersion: apps/v1
kind: Deployment
metadata:
  name: finance-ai-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: finance-ai
  template:
    metadata:
      labels:
        app: finance-ai
    spec:
      containers:
      - name: ai-container
        image: finance-ai:latest
        ports:
        - containerPort: 5000
        env:
        # Define your inputs here
        - name: AI_MODEL_VERSION
          value: "v1.2"
        - name: PROCESSING_MODE
          value: "batch"
