import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

# Import our custom modules
from src.dataset import load_tilda_dataset, get_tilda_transforms, BiasedDataset
from src.models import LeNet, AlexNet, ResNet, ScatteringCNN, NBDT
from src.train import train_model, distill_model
from src.evaluate import evaluate_model, evaluate_adversarial_robustness, evaluate_cw_attack, find_fair_thresholds
from src.utils import make_dirs, plot_curves

def create_synthetic_dataset(data_dir, num_samples=100, img_shape=(64, 64)):
    """
    Creates a dummy dataset with two classes (cauliflower and head_cabbage)
    if the TILDA dataset doesn't exist yet, to allow test runs.
    """
    classes = ['cauliflower', 'head_cabbage']
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
            print(f"Creating synthetic directory: {cls_dir}")
            for i in range(num_samples // 2):
                # Create random noise image (as 3-channel RGB PIL Image)
                # Let's add some simple visual pattern to distinguish them slightly
                img_array = np.random.randint(0, 256, (img_shape[0], img_shape[1], 3), dtype=np.uint8)
                if cls == 'cauliflower':
                    # Add a white box in the middle
                    img_array[20:44, 20:44, :] = 200
                else:
                    # Add a dark box in the middle
                    img_array[20:44, 20:44, :] = 50
                    
                img = Image.fromarray(img_array)
                img.save(os.path.join(cls_dir, f"img_{i}.png"))
    print("Synthetic dataset created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Project - TILDA Classification & Bias Study")
    parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'alexnet', 'resnet', 'scattering', 'nbdt'],
                        help="Model architecture to use.")
    parser.add_argument('--data_dir', type=str, default='data/tilda',
                        help="Path to TILDA dataset.")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate for SGD.")
    parser.add_argument('--p0', type=float, default=0.5,
                        help="Bias probability p0 = P(S=1 | y=0)")
    parser.add_argument('--p1', type=float, default=0.5,
                        help="Bias probability p1 = P(S=1 | y=1)")
    parser.add_argument('--img_size', type=int, default=64,
                        help="Resize input images to (img_size, img_size).")
    parser.add_argument('--binary', action='store_true',
                        help="Run in binary classification mode (Section 3).")
    parser.add_argument('--binary_classes', type=int, nargs=2, default=[0, 1],
                        help="The two class indices to use for binary classification.")
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'cw'],
                        help="Adversarial attack to evaluate (fgsm or cw).")
    parser.add_argument('--distill', action='store_true',
                        help="Train using defensive distillation.")
    parser.add_argument('--temperature', type=float, default=10.0,
                        help="Temperature for defensive distillation.")
    parser.add_argument('--mitigate', action='store_true',
                        help="Apply threshold-shifting fairness mitigation.")
    
    args = parser.parse_args()
    
    # Select Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # 1. Create or verify dataset
    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"Warning: Dataset directory '{args.data_dir}' not found or empty.")
        print("Initializing synthetic/dummy dataset for testing...")
        args.data_dir = 'data/dummy_tilda'
        create_synthetic_dataset(args.data_dir, num_samples=100, img_shape=(args.img_size, args.img_size))
        
    # Determine number of classes
    csv_file = os.path.join(args.data_dir, 'train.csv')
    is_kaggle = os.path.exists(csv_file)
    
    if is_kaggle:
        num_classes = 2 if args.binary else 8
        binary_classes = args.binary_classes if args.binary else None
        print(f"Detected Kaggle competition dataset. Mode: {'Binary (classes ' + str(binary_classes) + ')' if args.binary else 'Multi-class (8 classes)'}")
    else:
        num_classes = 2
        binary_classes = None
        print("Using dummy dataset (2 classes).")

    # 2. Load dataset
    print(f"Loading dataset from '{args.data_dir}' with bias parameters p0={args.p0}, p1={args.p1}...")
    dataset = load_tilda_dataset(
        args.data_dir, 
        image_size=(args.img_size, args.img_size), 
        p0=args.p0, 
        p1=args.p1, 
        augment=True,
        binary_classes=binary_classes
    )
    
    # Split into train/validation/test (e.g. 70% train, 15% val, 15% test)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset Split: {train_len} train, {val_len} val, {test_len} test samples.")
    
    # 3. Instantiate model
    # Note: the input channel dimension is the original channels (3 for RGB) + 1 for bias epsilon channel = 4
    in_channels = 4 
    
    print(f"Instantiating model '{args.model}' with {in_channels} input channels and {num_classes} output classes...")
    if args.model == 'lenet':
        model = LeNet(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
    elif args.model == 'alexnet':
        model = AlexNet(in_channels=in_channels, num_classes=num_classes)
    elif args.model == 'resnet':
        model = ResNet(in_channels=in_channels, num_classes=num_classes, pretrained=False)
    elif args.model == 'scattering':
        model = ScatteringCNN(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
    elif args.model == 'nbdt':
        model = NBDT(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
        
    # 4. Train model
    if args.distill:
        print("Starting distillation training...")
        # A. First train the teacher model on hard labels
        print("--- Step A: Training Teacher Model ---")
        teacher_model, training_time, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=args.epochs, 
            lr=args.lr, 
            device=device
        )
        
        # B. Instantiate student model of same architecture
        print("--- Step B: Distilling Student Model ---")
        if args.model == 'lenet':
            student_model = LeNet(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
        elif args.model == 'alexnet':
            student_model = AlexNet(in_channels=in_channels, num_classes=num_classes)
        elif args.model == 'resnet':
            student_model = ResNet(in_channels=in_channels, num_classes=num_classes, pretrained=False)
        elif args.model == 'scattering':
            student_model = ScatteringCNN(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
        elif args.model == 'nbdt':
            student_model = NBDT(in_channels=in_channels, num_classes=num_classes, image_size=(args.img_size, args.img_size))
            
        # C. Distill teacher into student
        trained_model, distill_time, distill_history = distill_model(
            teacher=teacher_model,
            student=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            device=device
        )
        history = distill_history
        training_time += distill_time
    else:
        print("Starting training...")
        trained_model, training_time, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=args.epochs, 
            lr=args.lr, 
            device=device
        )
    
    # Plot curves
    mode_str = "binary" if args.binary else "multiclass"
    if args.binary:
        if args.distill:
            curves_path = os.path.join("rapport", f"{args.model}_{mode_str}_p0_{args.p0}_p1_{args.p1}_distilled_learning_curves.png")
        else:
            curves_path = os.path.join("rapport", f"{args.model}_{mode_str}_p0_{args.p0}_p1_{args.p1}_learning_curves.png")
    else:
        if args.distill:
            curves_path = os.path.join("rapport", f"{args.model}_{mode_str}_distilled_learning_curves.png")
        else:
            curves_path = os.path.join("rapport", f"{args.model}_{mode_str}_learning_curves.png")
    plot_curves(history, save_path=curves_path)
    
    # 5. Evaluate on test set (Accuracy and DI metric)
    print("\nStarting evaluation on test set...")
    evaluate_model(trained_model, test_loader, device=device)
    
    # 6. Evaluate adversarial robustness
    print(f"\nStarting adversarial robustness evaluation ({args.attack.upper()})...")
    if args.attack == 'fgsm':
        evaluate_adversarial_robustness(trained_model, test_loader, epsilon=0.03, device=device)
    elif args.attack == 'cw':
        evaluate_cw_attack(trained_model, test_loader, c=1.0, steps=20, lr=0.01, device=device)
        
    # 7. Apply fairness mitigation if selected
    if args.binary and args.mitigate:
        print("\nStarting fairness mitigation (threshold shifting)...")
        find_fair_thresholds(trained_model, val_loader, test_loader, device=device)

if __name__ == "__main__":
    main()
