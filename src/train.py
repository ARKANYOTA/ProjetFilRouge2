import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=0.01, weight_decay=1e-4, device='cpu'):
    """
    Trains a model using mini-batch SGD and evaluates it on a validation set.
    Question 2.3 & 2.4.
    """
    model = model.to(device)
    # Question 2.3: SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    start_time = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct_train / total_train
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * correct_val / total_val
        
        # Save history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}: Train Loss={epoch_loss:.4f} | Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return model, total_time, history

def distill_model(teacher, student, train_loader, val_loader, epochs=10, lr=0.01, weight_decay=1e-4, temperature=10.0, device='cpu'):
    """
    Trains a student network using defensive distillation from a pre-trained teacher network.
    Paper: "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks" (Papernot et al., 2016)
    """
    import torch.nn.functional as F
    
    teacher = teacher.to(device)
    teacher.eval()
    
    student = student.to(device)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Validation criterion is standard CrossEntropy (evaluated at temperature 1)
    val_criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    start_time = time.time()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Get soft labels from teacher at temperature T
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                soft_labels = F.softmax(teacher_logits / temperature, dim=1)
                
            optimizer.zero_grad()
            
            # 2. Get student logits at temperature T
            student_logits = student(inputs)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # 3. Soft cross entropy loss
            # Scale by temperature^2 to keep gradients similar (standard distillation practice)
            loss = -(soft_labels * student_log_probs).sum(dim=1).mean() * (temperature ** 2)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Accuracy is still measured using student logits at temperature 1
            _, predicted = student_logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct_train / total_train
        
        # Validation loop (evaluated at temperature 1)
        student.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                loss = val_criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * correct_val / total_val
        
        # Save history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Distillation Epoch {epoch+1:02d}: Train Loss={epoch_loss:.4f} | Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), 'best_distilled_model.pth')
            
    total_time = time.time() - start_time
    print(f"\nDistillation completed in {total_time:.2f} seconds.")
    print(f"Best Distilled Validation Accuracy: {best_val_acc:.2f}%")
    
    return student, total_time, history

