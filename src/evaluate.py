import torch
import numpy as np

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates the model on test data, calculating accuracy, Disparate Impact (DI),
    Equalized Odds, and Equal Opportunity metrics.
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    # Track predictions split by label Y and bias variable S
    pred_1_S0 = 0
    total_S0 = 0
    pred_1_S1 = 0
    total_S1 = 0
    
    # Counts for Equalized Odds and Equal Opportunity
    total_Y0_S0 = 0
    pred1_Y0_S0 = 0
    total_Y0_S1 = 0
    pred1_Y0_S1 = 0
    
    total_Y1_S0 = 0
    pred1_Y1_S0 = 0
    total_Y1_S1 = 0
    pred1_Y1_S1 = 0
    
    with torch.no_grad():
        for inputs, labels, S_vars in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            preds_cpu = predicted.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            S_cpu = S_vars.numpy()
            
            for pred, label, s in zip(preds_cpu, labels_cpu, S_cpu):
                if s == 0:
                    total_S0 += 1
                    if pred == 1:
                        pred_1_S0 += 1
                elif s == 1:
                    total_S1 += 1
                    if pred == 1:
                        pred_1_S1 += 1
                
                # Equalized Odds counts
                if label == 0:
                    if s == 0:
                        total_Y0_S0 += 1
                        if pred == 1:
                            pred1_Y0_S0 += 1
                    elif s == 1:
                        total_Y0_S1 += 1
                        if pred == 1:
                            pred1_Y0_S1 += 1
                elif label == 1:
                    if s == 0:
                        total_Y1_S0 += 1
                        if pred == 1:
                            pred1_Y1_S0 += 1
                    elif s == 1:
                        total_Y1_S1 += 1
                        if pred == 1:
                            pred1_Y1_S1 += 1
                            
    accuracy = 100.0 * correct / total
    
    # Calculate probabilities for DI
    prob_1_S0 = pred_1_S0 / total_S0 if total_S0 > 0 else 0.0
    prob_1_S1 = pred_1_S1 / total_S1 if total_S1 > 0 else 0.0
    
    if prob_1_S1 == 0:
        di_metric = float('inf') if prob_1_S0 > 0 else 1.0
    else:
        di_metric = prob_1_S0 / prob_1_S1
        
    # Calculate False Positive Rates and True Positive Rates for Equalized Odds
    fpr_S0 = pred1_Y0_S0 / total_Y0_S0 if total_Y0_S0 > 0 else 0.0
    fpr_S1 = pred1_Y0_S1 / total_Y0_S1 if total_Y0_S1 > 0 else 0.0
    tpr_S0 = pred1_Y1_S0 / total_Y1_S0 if total_Y1_S0 > 0 else 0.0
    tpr_S1 = pred1_Y1_S1 / total_Y1_S1 if total_Y1_S1 > 0 else 0.0
    
    delta_fpr = abs(fpr_S0 - fpr_S1)
    delta_tpr = abs(tpr_S0 - tpr_S1)
    
    print(f"Evaluation Results:")
    print(f"  - Overall Accuracy: {accuracy:.2f}%")
    print(f"  - P(y_pred=1 | S=0): {prob_1_S0:.4f} ({pred_1_S0}/{total_S0})")
    print(f"  - P(y_pred=1 | S=1): {prob_1_S1:.4f} ({pred_1_S1}/{total_S1})")
    print(f"  - Disparate Impact (DI): {di_metric:.4f}")
    print(f"  - Fairness (Equalized Odds):")
    print(f"    * FPR (S=0): {fpr_S0:.4f} | FPR (S=1): {fpr_S1:.4f} | delta_FPR: {delta_fpr:.4f}")
    print(f"    * TPR (S=0): {tpr_S0:.4f} | TPR (S=1): {tpr_S1:.4f} | delta_TPR (Equal Opp): {delta_tpr:.4f}")
    
    return accuracy, di_metric, delta_fpr, delta_tpr

def evaluate_adversarial_robustness(model, test_loader, epsilon=0.05, device='cpu'):
    """
    Evaluates model robustness under FGSM adversarial attack.
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for inputs, labels, _ in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        perturbed_inputs = inputs + epsilon * data_grad.sign()
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        with torch.no_grad():
            outputs_adv = model(perturbed_inputs)
            _, predicted = outputs_adv.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    adv_accuracy = 100.0 * correct / total
    print(f"Adversarial Evaluation (FGSM, epsilon={epsilon}):")
    print(f"  - Adversarial Accuracy: {adv_accuracy:.2f}%")
    return adv_accuracy

def evaluate_cw_attack(model, test_loader, c=1.0, kappa=0.0, steps=20, lr=0.01, device='cpu'):
    """
    Evaluates model robustness under Carlini-Wagner L_2 adversarial attack.
    Paper: "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
    This is an optimization-based attack using Adam.
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    for inputs, labels, _ in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs_scaled = (inputs * 2.0 - 1.0) * 0.999999
        w_init = torch.arctanh(inputs_scaled)
        
        w = w_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            adv_inputs = 0.5 * (torch.tanh(w) + 1.0)
            l2_loss = torch.sum((adv_inputs - inputs) ** 2, dim=[1, 2, 3])
            
            logits = model(adv_inputs)
            correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            
            mask = torch.ones_like(logits)
            mask.scatter_(1, labels.unsqueeze(1), 0)
            other_logits = logits * mask - (1.0 - mask) * 1e9
            max_other_logits = other_logits.max(dim=1)[0]
            
            f_loss = torch.clamp(correct_logits - max_other_logits, min=-kappa)
            loss = torch.mean(l2_loss + c * f_loss)
            
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_adv_inputs = 0.5 * (torch.tanh(w) + 1.0)
            outputs_adv = model(final_adv_inputs)
            _, predicted = outputs_adv.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    adv_accuracy = 100.0 * correct / total
    print(f"Adversarial Evaluation (C&W L_2, c={c}, steps={steps}):")
    print(f"  - Adversarial Accuracy: {adv_accuracy:.2f}%")
    return adv_accuracy

def find_fair_thresholds(model, val_loader, test_loader, device='cpu'):
    """
    Finds group-specific classification thresholds tau_0, tau_1 on validation data
    to satisfy demographic parity / Disparate Impact constraint (0.8 <= DI <= 1.25)
    and maximizes validation accuracy. Then applies them to the test set.
    """
    model.eval()
    model = model.to(device)
    
    val_probs = []
    val_labels = []
    val_S = []
    
    with torch.no_grad():
        for inputs, labels, S_vars in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.numpy())
            val_S.extend(S_vars.numpy())
            
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    val_S = np.array(val_S)
    
    thresholds = np.linspace(0.05, 0.95, 19)
    best_acc = 0.0
    best_thresholds = (0.5, 0.5)
    best_di = 1.0
    found_fair = False
    
    for t0 in thresholds:
        for t1 in thresholds:
            preds = np.zeros_like(val_probs)
            preds[val_S == 0] = (val_probs[val_S == 0] > t0).astype(int)
            preds[val_S == 1] = (val_probs[val_S == 1] > t1).astype(int)
            
            acc = np.mean(preds == val_labels)
            
            pred_1_S0 = np.sum((preds == 1) & (val_S == 0))
            total_S0 = np.sum(val_S == 0)
            pred_1_S1 = np.sum((preds == 1) & (val_S == 1))
            total_S1 = np.sum(val_S == 1)
            
            p0 = pred_1_S0 / total_S0 if total_S0 > 0 else 0.0
            p1 = pred_1_S1 / total_S1 if total_S1 > 0 else 0.0
            
            if p1 == 0:
                di = float('inf') if p0 > 0 else 1.0
            else:
                di = p0 / p1
                
            is_fair = (0.8 <= di <= 1.25)
            
            if is_fair:
                if not found_fair or acc > best_acc:
                    best_acc = acc
                    best_thresholds = (t0, t1)
                    best_di = di
                    found_fair = True
            else:
                if not found_fair and abs(di - 1.0) < abs(best_di - 1.0):
                    best_thresholds = (t0, t1)
                    best_di = di
                    best_acc = acc
                    
    print(f"\nFair Threshold Selection on Validation Set:")
    print(f"  - Selected threshold for S=0: {best_thresholds[0]:.2f}")
    print(f"  - Selected threshold for S=1: {best_thresholds[1]:.2f}")
    print(f"  - Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"  - Validation Disparate Impact (DI): {best_di:.4f}")
    
    test_probs = []
    test_labels = []
    test_S = []
    
    with torch.no_grad():
        for inputs, labels, S_vars in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_S.extend(S_vars.numpy())
            
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_S = np.array(test_S)
    
    t0, t1 = best_thresholds
    test_preds = np.zeros_like(test_probs)
    test_preds[test_S == 0] = (test_probs[test_S == 0] > t0).astype(int)
    test_preds[test_S == 1] = (test_probs[test_S == 1] > t1).astype(int)
    
    test_acc = np.mean(test_preds == test_labels)
    
    pred_1_S0 = np.sum((test_preds == 1) & (test_S == 0))
    total_S0 = np.sum(test_S == 0)
    pred_1_S1 = np.sum((test_preds == 1) & (test_S == 1))
    total_S1 = np.sum(test_S == 1)
    
    p0 = pred_1_S0 / total_S0 if total_S0 > 0 else 0.0
    p1 = pred_1_S1 / total_S1 if total_S1 > 0 else 0.0
    
    if p1 == 0:
        test_di = float('inf') if p0 > 0 else 1.0
    else:
        test_di = p0 / p1
        
    print(f"Fairness Mitigation Results on Test Set:")
    print(f"  - Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  - Test Disparate Impact (DI): {test_di:.4f}")
    
    return test_acc * 100.0, test_di

