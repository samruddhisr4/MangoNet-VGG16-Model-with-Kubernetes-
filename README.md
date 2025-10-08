# MangoNet: VGG16-based Mango Classification with Kubernetes Deployment

**MangoNet** is a neural network project using the **VGG16** convolutional model to classify mango images. The system is containerized and deployed using **Kubernetes** to enable scalable inference.

---

## 🚀 Features

* Transfer learning using a VGG16 backbone, fine-tuned on mango datasets
* REST API for image inference
* Containerization with Docker
* Deployment orchestration via Kubernetes (pods, services, autoscaling)
* Logging, monitoring, and scale-out capabilities

---

## 📂 Project Structure

```
├── docker/  
│   └── Dockerfile          # Builds the inference container  
│  
├── k8s/  
│   ├── deployment.yaml     # Kubernetes deployment definition  
│   ├── service.yaml        # Kubernetes service (ClusterIP / LoadBalancer)  
│   └── hpa.yaml             # Horizontal Pod Autoscaler config  
│  
├── model/  
│   ├── weights.h5          # Trained model weights  
│   └── vgg_mango.py        # Model definition & utilities  
│  
├── src/  
│   ├── app.py               # FastAPI / Flask app for inference  
│   ├── requirements.txt     # Python dependencies  
│   └── utils.py             # Preprocessing, helper functions  
│  
├── tests/  
│   └── test_inference.py    # Unit tests for API & model  
│  
└── README.md  
```

---

## 🛠 Setup & Usage

### 1. Model Training & Preparation

* Train or fine-tune VGG16 on your mango dataset.
* Save the model as `weights.h5`.
* (Optional) Perform data augmentation, hyperparameter search, etc.

### 2. Build Docker Image

```bash
cd docker  
docker build -t mangonet:v1 .
```

### 3. Kubernetes Deployment

* Apply the deployment, service, and autoscaling manifests:

  ```bash
  kubectl apply -f k8s/deployment.yaml  
  kubectl apply -f k8s/service.yaml  
  kubectl apply -f k8s/hpa.yaml  
  ```
* Monitor pod status:

  ```bash
  kubectl get pods  
  kubectl get svc  
  ```

### 4. Inference

* Send a `POST` via `curl` or HTTP client to the API endpoint:

  ```bash
  curl -X POST http://<service_ip>/predict \
    -F "image=@mango.jpg"
  ```
* Response JSON: predicted class and confidence.

---

## 📊 Model Architecture & Approach

* **Base model**: VGG16 pretrained on ImageNet (top layers removed)
* **Custom head**: Dense + Dropout + Softmax for mango classes
* **Training strategy**:

  1. Freeze early VGG layers, train top layers
  2. Unfreeze select deeper layers and fine-tune
* **Input size**: 224 × 224 RGB
* **Preprocessing**: Resize, normalize (rescale / mean subtraction)
* **Loss / metrics**: Categorical crossentropy, accuracy, F1 etc.

---

## 🧪 Testing & Validation

* Unit tests in `tests/` ensure that:

  * The model loads successfully
  * The API returns results for valid requests
  * Invalid inputs are handled gracefully
* Evaluate performance on held-out test data (accuracy, confusion matrix)

---

## 📈 Scalability & Kubernetes Features

* **Horizontal Pod Autoscaler** (HPA) scales inference pods based on CPU/memory or request latency
* **Load Balancing** through Kubernetes Service
* Easy to roll out updates by applying new Docker image tags

---

## 🧮 Results & Performance

| Metric            | Value      |
| ----------------- | ---------- |
| Test Accuracy     | e.g. 92.5% |
| Inference Latency | ~100 ms    |
| Model Size        | ~100 MB    |

(These numbers are illustrative — use your actual metrics.)

---

## ✅ Prerequisites

* Docker & Kubernetes installed
* `kubectl` access to a cluster
* Python 3.8+
* GPU (optional, for training)
* Dataset of mango images with labels

---

## 💡 Future Improvements

* Add support for more mango varieties
* Add model versioning / CI/CD
* Incorporate ensemble models
* Monitor model drift, retraining pipeline
* Add authentication, RBAC for API
* Use GPU-based autoscaling or serverless inference

---

## 📚 References

* **VGG16 architecture details** — small 3×3 filters, deep convolutional layers, ~138 million parameters ([Medium][1])
* **Transfer learning with Keras / VGG16** (freeze, fine-tune) ([Medium][2])
* **Mango image classification with VGG16** achieves ~92.50% accuracy in published work ([Journal of Social Science][3])

---

## 🏷 License & Credits

* MIT / Apache / [choose your license]
* Developed by **samruddhisr4**
* If you use any external code or data, acknowledge appropriately

---
