import torch
import numpy as np
from sklearn.metrics import f1_score


def get_mrr(prediction, targets):
    """
    Calculates the MRR score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)

    return torch.sum(rranks).cpu().numpy()


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float().cpu().numpy()

    not_considered_idx = ranks > k
    ndcg = 1 / np.log2(ranks + 1)
    ndcg[not_considered_idx] = 0

    return np.sum(ndcg)


def calculate_correct_total_prediction(logits, true_y):

    # top_ = torch.eq(torch.argmax(logits, dim=-1), true_y).sum().cpu().numpy()
    top1 = []
    result_ls = []
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        prediction = torch.topk(logits, k=k, dim=-1).indices
        # f1 score
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()
            # f1 = f1_score(true_y.cpu(), prediction.cpu(), average="weighted")

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        # top_k = np.sum([curr_y in pred for pred, curr_y in zip(prediction, true_y)])
        result_ls.append(top_k)
    # f1 score
    # result_ls.append(f1)
    # rr
    result_ls.append(get_mrr(logits, true_y))
    # ndcg
    result_ls.append(get_ndcg(logits, true_y))

    # total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1


def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "ndcg": return_dict["ndcg"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100
    perf["ndcg"] = perf["ndcg"] / perf["total"] * 100

    return perf


def evaluate_model(model, dataloader, device, hierarchy_map=None, use_hierarchy=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The hierarchical S2 model
        dataloader: DataLoader for evaluation
        device: torch device
        hierarchy_map: Dictionary with hierarchical mappings
        use_hierarchy: Whether to use hierarchical restrictions
    
    Returns:
        Dictionary with performance metrics for each level
    """
    model.eval()
    
    metrics = {
        'l11': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
        'l13': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
        'l14': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
                'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
        'X': {'correct@1': 0, 'correct@3': 0, 'correct@5': 0, 'correct@10': 0, 
              'rr': 0, 'ndcg': 0, 'f1': 0, 'total': 0},
    }
    
    with torch.no_grad():
        for batch in dataloader:
            l11_seq = batch['l11_seq'].to(device)
            l13_seq = batch['l13_seq'].to(device)
            l14_seq = batch['l14_seq'].to(device)
            X_seq = batch['X_seq'].to(device)
            user_seq = batch['user_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            l11_y = batch['l11_y'].to(device)
            l13_y = batch['l13_y'].to(device)
            l14_y = batch['l14_y'].to(device)
            X_y = batch['X_y'].to(device)
            
            # Forward pass
            outputs = model(l11_seq, l13_seq, l14_seq, X_seq, user_seq, padding_mask)
            
            logits_l11 = outputs['logits_l11']
            logits_l13 = outputs['logits_l13']
            logits_l14 = outputs['logits_l14']
            logits_X = outputs['logits_X']
            
            # Evaluate each level
            result, _, _ = calculate_correct_total_prediction(logits_l11, l11_y)
            metrics['l11']['correct@1'] += result[0]
            metrics['l11']['correct@3'] += result[1]
            metrics['l11']['correct@5'] += result[2]
            metrics['l11']['correct@10'] += result[3]
            metrics['l11']['rr'] += result[4]
            metrics['l11']['ndcg'] += result[5]
            metrics['l11']['total'] += result[6]
            
            result, _, _ = calculate_correct_total_prediction(logits_l13, l13_y)
            metrics['l13']['correct@1'] += result[0]
            metrics['l13']['correct@3'] += result[1]
            metrics['l13']['correct@5'] += result[2]
            metrics['l13']['correct@10'] += result[3]
            metrics['l13']['rr'] += result[4]
            metrics['l13']['ndcg'] += result[5]
            metrics['l13']['total'] += result[6]
            
            result, _, _ = calculate_correct_total_prediction(logits_l14, l14_y)
            metrics['l14']['correct@1'] += result[0]
            metrics['l14']['correct@3'] += result[1]
            metrics['l14']['correct@5'] += result[2]
            metrics['l14']['correct@10'] += result[3]
            metrics['l14']['rr'] += result[4]
            metrics['l14']['ndcg'] += result[5]
            metrics['l14']['total'] += result[6]
            
            result, _, _ = calculate_correct_total_prediction(logits_X, X_y)
            metrics['X']['correct@1'] += result[0]
            metrics['X']['correct@3'] += result[1]
            metrics['X']['correct@5'] += result[2]
            metrics['X']['correct@10'] += result[3]
            metrics['X']['rr'] += result[4]
            metrics['X']['ndcg'] += result[5]
            metrics['X']['total'] += result[6]
    
    # Convert to performance dict
    perf = {}
    for level in ['l11', 'l13', 'l14', 'X']:
        perf[level] = get_performance_dict(metrics[level])
    
    return perf
