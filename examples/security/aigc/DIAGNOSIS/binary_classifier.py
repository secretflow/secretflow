import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
from PIL import Image
import argparse
import glob
import os
import numpy as np
import time
from collections import Counter

def pil_loader(path: str) -> Image.Image:
    """Loads an image as RGB using PIL."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def setup_seed(seed=42):
    """Sets seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom Dataset
class TwoClassesImageFolder(Dataset):
    """Dataset to handle two-class image classification."""
    def __init__(self, ori_dir=None, coated_dir=None, transform=None):
        self.labels = []
        self.paths = []

        all_ori_paths = glob.glob(os.path.join(ori_dir, "*.png"))
        all_coated_paths = glob.glob(os.path.join(coated_dir, "*.png"))

        for ori_path in all_ori_paths:
            self.labels.append(0)
            self.paths.append(ori_path)

        for coated_path in all_coated_paths:
            self.labels.append(1)
            self.paths.append(coated_path)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        target = self.labels[idx]
        sample = pil_loader(self.paths[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def analyze_dataset_distribution(train_set, test_set, result_file_path):
    """
    Analyze the distribution of labels in training and test datasets
    and save results to a file.
    
    Args:
        train_set: Training dataset
        test_set: Test dataset
        result_file_path: Path to save the results
    """
    # Count labels in training set
    train_labels = []
    for idx in range(len(train_set)):
        _, label = train_set[idx]
        train_labels.append(label)
    train_counter = Counter(train_labels)
    
    # Count labels in test set
    test_labels = []
    for idx in range(len(test_set)):
        _, label = test_set[idx]
        test_labels.append(label)
    test_counter = Counter(test_labels)
    
    # Create result directory if it doesn't exist
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    
    # Save results to file
    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write("Dataset Distribution Analysis\n")
        f.write("==========================\n\n")
        
        f.write("Training Set Distribution:\n")
        f.write(f"Label 0 count: {train_counter[0]}\n")
        f.write(f"Label 1 count: {train_counter[1]}\n")
        f.write(f"Total training samples: {len(train_set)}\n\n")
        
        f.write("Test Set Distribution:\n")
        f.write(f"Label 0 count: {test_counter[0]}\n")
        f.write(f"Label 1 count: {test_counter[1]}\n")
        f.write(f"Total test samples: {len(test_set)}\n")
        
    return train_counter, test_counter

def train(model, train_loader, optimizer, criterion, epoch, result_file_path):
    model.train()
    start_time = time.time()
    
    # Open file in append mode
    with open(result_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\nTraining Epoch {epoch}\n")
        f.write("====================\n")
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 25 == 0:
            log_message = f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} " \
                         f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            print(log_message)
            with open(result_file_path, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

    epoch_time = time.time() - start_time
    time_message = f"Time for epoch {epoch}: {epoch_time:.2f}s"
    print(time_message)
    with open(result_file_path, 'a', encoding='utf-8') as f:
        f.write(time_message + '\n\n')

def test(model, test_loader, criterion, num_classes, result_file_path):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = [0] * num_classes
    total_class = [0] * num_classes

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            for i in range(data.size(0)):
                total_class[target[i].item()] += 1
                if target[i].item() == pred[i].item():
                    correct_class[target[i].item()] += 1

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    with open(result_file_path, 'a', encoding='utf-8') as f:
        f.write("\nTest Results\n")
        f.write("============\n")
        f.write(f"Test Set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
                f"({accuracy:.2f}%)\n\n")
        
        f.write("Per-Class Performance:\n")
        for i, (count, total) in enumerate(zip(correct_class, total_class)):
            class_accuracy = 100. * count / total if total > 0 else 0
            result = f"Class {i}: {count}/{total} ({class_accuracy:.2f}%)"
            print(result)
            f.write(result + '\n')
        f.write('\n')

    return accuracy


def inspect_memorization_strength(model, args, transform):
    """
    Inspects the memorization strength of the model by analyzing 
    its performance on generated images (normal or triggered),
    and saves the results to a result.txt file.

    Args:
        model: Trained PyTorch model.
        args: Parsed command-line arguments.
        transform: Transformation to apply to the images.
    """
    model.eval()
    memorization_list = []

    with torch.no_grad():
        for i in range(50):
            # Determine the file path based on whether it's conditional or unconditional
            if args.trigger_conditioned:
                file_path = f"{args.generated_inspected_dir}/triggered_{i}.png"
            else:
                file_path = f"{args.generated_inspected_dir}/normal_{i}.png"
            print(file_path)
            # Load and preprocess the image
            data = Image.open(file_path)
            data = transform(data).unsqueeze(0).cuda()
            
            # Pass through the model
            output = model(data)
            print(F.softmax(output, dim=1))

            # Record the prediction
            memorization_list.append(output.argmax(dim=1).item())

    # Compute memorization strength
    memorization_tensor = torch.tensor(memorization_list)
    memorization_acc = (memorization_tensor == 1).float().mean().item()

    # Determine malicious/benign model status
    is_malicious = memorization_acc > 0.75
    status = "Malicious Model!" if is_malicious else "Benign Model"

    # Print results to console
    print(f"Memorization Strength: {memorization_acc:.2f}")
    print(status)

    # Save results to result.txt
    result_file_path = os.path.join(args.generated_inspected_dir, "result.txt")
    with open(result_file_path, "a") as result_file:
        result_file.write(f"Memorization Strength: {memorization_acc:.2f}\n")
        result_file.write(f"Model Status: {status}\n")
    
    print(f"Results saved to {result_file_path}")



# FID Calculation
def calculate_fid(args):
    def fid_preprocess_image(image):
        """Preprocess image for FID computation."""
        image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        return torchvision.transforms.functional.center_crop(image, (256, 256))

    real_paths = glob.glob(os.path.join(args.ori_dir, "*.png"))
    fake_paths = glob.glob(os.path.join(args.generated_inspected_dir, "*.png"))

    fake_paths = [path for path in fake_paths if "normal" in os.path.basename(path)]

    real_images = [np.array(Image.open(path).convert("RGB")) for path in real_paths]
    fake_images = [np.array(Image.open(path).convert("RGB")) for path in fake_paths]

    real_images = torch.cat([fid_preprocess_image(img) for img in real_images])
    fake_images = torch.cat([fid_preprocess_image(img) for img in fake_images])

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    fid_score = fid.compute()
    # Save FID result to result.txt
    result_file_path = os.path.join(args.generated_inspected_dir, "result.txt")
    with open(result_file_path, "a") as result_file:  # Open in append mode to avoid overwriting
        result_file.write(f"FID: {fid_score:.4f}\n")

    print(f"FID: {fid_score:.4f}")
    print(f"FID score saved to {result_file_path}")


def save_model(model, save_path, dataset_method):
    """Saves the model's state_dict."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_save_path = os.path.join(save_path, f"{dataset_method}_epoch_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a ResNet-based classifier.")
    parser.add_argument("--ori_dir", type=str, required=True, help="Directory containing original images.")
    parser.add_argument("--coated_dir", type=str, required=True, help="Directory containing coated images.")
    parser.add_argument("--generated_inspected_dir", type=str, required=True, help="Directory containing generated images for inspection.")
    parser.add_argument("--trigger_conditioned", action="store_true", help="Use trigger-conditioned images for inspection.")
    parser.add_argument("--dataset_method", type=str, required=True, help="Name of the dataset/method.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    setup_seed()

    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TwoClassesImageFolder(args.ori_dir, args.coated_dir, transform_train)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    test_set.dataset.transform = transform_test

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model setup
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Two-class classification

    # Use dataset_method for checkpoint storage
    checkpoint_path = "./bc_checkpoint"
    checkpoint_file = os.path.join(checkpoint_path, f"{args.dataset_method}_epoch_final.pth")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file))
        print(f"Loaded model checkpoint from '{checkpoint_file}'")
        args.epochs = 0
    else:
        print(f"No checkpoint found at '{checkpoint_file}'. Starting training from scratch.")

    model.cuda()

    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    result_file_path = os.path.join(args.generated_inspected_dir, "result.txt")
    # Training and evaluation
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, result_file_path=result_file_path)
        test(model, test_loader, criterion, num_classes=2, result_file_path=result_file_path)

    # Save the final model
    save_model(model, checkpoint_path, args.dataset_method)

    # Inspect and calculate FID
    inspect_memorization_strength(model, args, transform_test)
    calculate_fid(args)

if __name__ == "__main__":
    main()
