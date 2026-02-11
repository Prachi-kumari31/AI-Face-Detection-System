import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ================= MODEL CACHE =================
_cached_model = None
_cached_device = None


class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeCNN, self).__init__()

        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model():
    global _cached_model, _cached_device

    if _cached_model is None:
        print("[DEBUG] Loading model for first time...", file=sys.stderr)

        # Device selection
        _cached_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEBUG] Using device: {_cached_device}", file=sys.stderr)

        # Model init
        model = DeepfakeCNN(num_classes=2)

        # Model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "..", "models", "deepfake_detector_cnn.pth")

        print(f"[DEBUG] Model path: {model_path}", file=sys.stderr)
        print(f"[DEBUG] Model exists: {os.path.exists(model_path)}", file=sys.stderr)

        # Load weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=_cached_device)

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print("[DEBUG] ✅ Model weights loaded from checkpoint", file=sys.stderr)

                    if "accuracy" in checkpoint:
                        print(
                            f"[DEBUG] Trained accuracy: {checkpoint['accuracy']:.2f}%",
                            file=sys.stderr
                        )
                else:
                    model.load_state_dict(checkpoint)
                    print("[DEBUG] ✅ Model weights loaded successfully", file=sys.stderr)

            except Exception as e:
                print(f"[DEBUG] ⚠️ Could not load model weights: {e}", file=sys.stderr)
                print("[DEBUG] ⚠️ Using RANDOM weights", file=sys.stderr)
        else:
            print(f"[DEBUG] ❌ Model file not found at: {model_path}", file=sys.stderr)
            print("[DEBUG] ⚠️ Using RANDOM weights", file=sys.stderr)

        model.to(_cached_device)
        model.eval()

        _cached_model = model
        print("[DEBUG] Model cached and ready", file=sys.stderr)

    return _cached_model, _cached_device


def predict_image(image_path):
    try:
        model, device = get_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[DEBUG] Loading image: {image_path}", file=sys.stderr)
        print(f"[DEBUG] Image exists: {os.path.exists(image_path)}", file=sys.stderr)

        image = Image.open(image_path).convert("RGB")
        print(f"[DEBUG] Original image size: {image.size}", file=sys.stderr)

        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"[DEBUG] Tensor shape: {image_tensor.shape}", file=sys.stderr)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

            labels = ["FAKE", "REAL"]
            prediction_label = labels[predicted_class]
            confidence = probabilities[0][predicted_class].item() * 100

            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100

            return {
                "success": True,
                "prediction": prediction_label,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "FAKE": round(fake_prob, 2),
                    "REAL": round(real_prob, 2)
                }
            }

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "No image path provided"
        }), flush=True)
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)

    print(json.dumps(result), flush=True)
    sys.exit(0)
