from torch.nn import CosineSimilarity
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import print_train_info
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(  
        F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss


def instance_bce(logits, labels):
    # standard cross-entropy loss
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)

    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss


# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def get_bce_loss(opt, logits, a):
    if opt.loss_fn == 'ml_loss':
        bce_loss = instance_bce_with_logits(
            logits, a, reduction='mean')
    elif opt.loss_fn == 'ce_loss':
        bce_loss = instance_bce(logits, a)

    return bce_loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def criterion(out_1, out_2, tau_plus=0.1, batch_size=64, beta=1.0, estimator='easy', temperature=0.5):
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp*neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e**(-1 / temperature))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception(
            'Invalid estimator selected. Please use any of [hard, easy]')
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()
    return loss


class Contrastive_loss(nn.Module):
    def __init__(self, tao):
        super(Contrastive_loss, self).__init__()
        self.sim = CosineSimilarity(dim=-1)
        self.tao = tao

    def forward(self, fea, pos_fea, neg_fea):
        fea = F.normalize(fea, dim=1)
        pos_fea = F.normalize(pos_fea, dim=1)
        neg_fea = F.normalize(neg_fea, dim=1)

        pos_sim = self.sim(fea, pos_fea)
        neg_sim = self.sim(fea, neg_fea)

        logits = torch.exp(pos_sim / self.tao) / \
            (torch.exp(pos_sim / self.tao) + torch.exp(neg_sim / self.tao))
        loss = (-1.0 * torch.log(logits))

        return loss.mean()


def train(model, train_loader, eval_loader, opt, qid2type):

    # with open('util/qid2type_%s.json' % opt.task, 'r') as f:
    #     qid2type = json.load(f)
    utils.create_dir(opt.output)  

    # optim = torch.optim.Adamax(model.parameters())
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)

    logger = utils.Logger(os.path.join(opt.output, 'log.txt'))
    print_train_info(opt, logger, __file__)

    # load snapshot
    best_eval_score = 0
    best_eval_epoch = 0
    eps = 1e-7

    for param_group in optim.param_groups:  
        param_group['lr'] = opt.learning_rate

    contrastive_loss = Contrastive_loss(opt.tao).cuda()

    if opt.checkpoint_path is not None:
        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1
        best_eval_score = model_data['best_eval_score']
        best_eval_epoch = model_data['best_eval_epoch']
        logger.write("epoch:%d\tbest_eval_epoch:%d\tbest_eval_score:%.2f" %
                     (model_data['epoch']+1, best_eval_epoch, 100*best_eval_score))

    scheduler = MultiStepLR(
        optim, milestones=[10, 15, 20, 25, 30, 35], gamma=0.5)
    scheduler.last_epoch = opt.s_epoch

    for epoch in range(opt.s_epoch, opt.num_epochs):
        total_loss = 0
        total_bce_loss = 0
        total_con_loss = 0
        total_p_loss = 0
        total_self_loss = 0
        total_re_loss = 0
        train_score_re = 0
        train_score_pos = 0
        train_score_p = 0
        train_score_neg = 0
        total_norm = 0
        count_norm = 0

        total_cl_loss = 0

        t = time.time()
        N = len(train_loader.dataset)

        for i, (v, box, q, a, qids, bias,  hint_score, Shuffling_q, Removal_q, positive_q, bias_or_not) in tqdm(enumerate(train_loader), ncols=100, desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            v = v.cuda()
            box = box.cuda()
            q = q.cuda()
            a = a.cuda()
            bias = bias.cuda()
            qids = qids.detach().cpu().int().numpy()
            hint_score = hint_score.cuda()
            neg_gta = torch.zeros_like(a)  

            Shuffling_q = Shuffling_q.cuda()
            Removal_q = Removal_q.cuda()
            positive_q = positive_q.cuda()
            bias_or_not = bias_or_not.cuda()


            if opt.mode in ['updn', 'san', 'ban']:

                out = model(q, v)
                bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                loss = bce_loss_pos
            elif opt.mode in ['updn-ssl', 'san-ssl', 'ban-ssl']:
                if epoch < opt.pretrain_epoches: 
                    out = model(q, v)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    loss = bce_loss_pos
                else:
                    out = model(q, v, self_sup=True)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    self_loss_v = compute_self_loss(out['logits_neg'], a)
                    self_loss = self_loss_v
                    if opt.bias_filter:
                        self_loss = (self_loss * bias_or_not).sum() / \
                            (bias_or_not.sum() + eps)
                    loss = bce_loss_pos + opt.self_loss_weight * self_loss

            elif opt.mode in ['updn-re']:
                if epoch < opt.pretrain_epoches:
                    out = model(q, v)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    loss = bce_loss_pos
                else:
                    out = model(q, v, self_sup=True, positive_q=positive_q)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    self_loss_v = compute_self_loss(out['logits_neg'], a)
                    self_loss_q = compute_self_loss(out['logits_neg_q'], a)

                    self_loss = self_loss_v + opt.self_loss_q * self_loss_q

                    v_mask = torch.zeros(v.shape[0], 36).cuda()
                    hint_sort, hint_ind = hint_score.sort(1, descending=True)
                    v_ind = hint_ind[:, :opt.top_hint]  # 前

                    v_grad = out['att_gv'].squeeze(-1).gather(1, v_ind)

                    v_grad_ind = v_grad.sort(1, descending=True)[
                        1][:, :opt.topv]
                    v_star = v_ind.gather(1, v_grad_ind)
                    v_mask.scatter_(1, v_star, 1)  

                    v_mask = 1 - v_mask  
                    re_out = model(q, v, v_mask=v_mask)

                    re_loss = get_bce_loss(opt, re_out['logits_pos'], neg_gta)

                    if opt.bias_filter:
                        self_loss = (self_loss * bias_or_not).sum() / \
                            (bias_or_not.sum() + eps)
                        re_loss = (re_loss * bias_or_not).sum() / \
                            (bias_or_not.sum() + eps)

                    loss = bce_loss_pos + opt.lamda * re_loss + opt.self_loss_weight * self_loss

            elif opt.mode in ['updn-cl']:
                if epoch < opt.pretrain_epoches:  
                    out = model(q, v, positive_q=positive_q)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    bce_loss_positive = get_bce_loss(
                        opt, out['logits_positive'], a)  

                    loss = bce_loss_pos + bce_loss_positive
                else:
                    out = model(q, v, self_sup=True, positive_q=positive_q)
                    bce_loss_pos = get_bce_loss(opt, out['logits_pos'], a)
                    self_loss_v = compute_self_loss(out['logits_neg'], a)
                    self_loss_q = compute_self_loss(out['logits_neg_q'], a)

                    self_loss = self_loss_v + opt.self_loss_q * self_loss_q

                    bce_loss_positive = get_bce_loss(
                        opt, out['logits_positive'], a)
                    cl_loss = criterion(out['norm_repr_ori'], out['norm_repr_positive'],
                                        tau_plus=0.1, beta=1, batch_size=q.size(0),  estimator='easy', temperature=0.5)

                    con_loss = contrastive_loss(
                        out['norm_repr_ori'], out['norm_repr_positive'], out['norm_repr_neg'])

                    if opt.bias_filter:
                        self_loss = (self_loss * bias_or_not).sum() / \
                            (bias_or_not.sum() + eps)

                    loss = bce_loss_pos + opt.self_loss_weight * \
                        self_loss + bce_loss_positive

                    contrast_loss = 0.4 * cl_loss + 1 * con_loss
                    loss += contrast_loss

            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(
                model.parameters(), opt.grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            score_pos = compute_score_with_logits(
                out['logits_pos'], a.data).sum()
            train_score_pos += score_pos.item()
            total_loss += loss.item() * v.size(0)
            total_bce_loss += bce_loss_pos.item() * v.size(0)

            if opt.mode in ['updn-ssl', 'san-ssl', 'ban-ssl', 'updn-re', 'updn-cl']:
                if epoch < opt.pretrain_epoches:
                    total_self_loss = 0
                    train_score_neg = 0
                else:
                    score_neg = compute_score_with_logits(
                        out['logits_neg'], a.data).sum()
                    train_score_neg += score_neg.item()
                    total_self_loss += self_loss.item() * v.size(0)

            if opt.mode in ['updn-re']:
                if epoch < opt.pretrain_epoches:
                    pass
                else:
                    score_re = compute_score_with_logits(
                        re_out['logits_pos'], a.data).sum()
                    train_score_re += score_re.item()
                    total_re_loss += re_loss.item() * v.size(0)

            if opt.mode in ['updn-cl']:

                score_p = compute_score_with_logits(
                    out['logits_positive'], a.data).sum()
                train_score_p += score_p.item()
                total_p_loss += bce_loss_positive.item() * v.size(0)

                if epoch < opt.pretrain_epoches:
                    total_cl_loss = 0
                    total_con_loss = 0
                else:
                    total_con_loss += con_loss.item() * v.size(0)
                    total_cl_loss += cl_loss.item() * v.size(0)

            if i != 0 and i % ((len(train_loader)//3)+10) == 0:
                sout = '\ntraing: %d/%d, train_loss: %.6f, bce_loss: %.6f,pos_train_acc: %.6f, ' % (i, len(
                    train_loader), total_loss / (i * v.size(0)), total_bce_loss / (i * v.size(0)), 100 * train_score_pos / (i * v.size(0)))

                if opt.mode in ['updn-cl']:
                    sout += 'cl_loss: %.6f, p_loss: %.6f, p_train_acc: %.6f, con_loss: %.6f, ' % (
                        total_cl_loss / (i * v.size(0)),
                        total_p_loss / (i * v.size(0)),
                        100 * train_score_p / (i * v.size(0)),
                        total_con_loss / (i * v.size(0)),
                    )

                if opt.mode in ['updn-re']:
                    sout += 're_loss: %.6f, re_train_acc: %.6f, ' % (total_re_loss / (
                        i * v.size(0)), 100 * train_score_re / (i * v.size(0)),)

                if opt.mode in ['updn-ssl', 'san-ssl', 'ban-ssl', 'updn-re', 'updn-cl']:
                    sout += 'self_loss: %.6f, neg_train_acc: %.6f, ' % (total_self_loss / (
                        i * v.size(0)), 100 * train_score_neg / (i * v.size(0)),)

                logger.write(sout)

        scheduler.step()
        total_loss /= N
        total_bce_loss /= N
        total_self_loss /= N
        train_score_pos = 100 * train_score_pos / N


        if None != eval_loader:
            print("Starting eval...")
            model.train(False)
            eval_out = evaluate(model, eval_loader, qid2type, opt)

            model.train(True)

        logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time() - t))
        logger.write(
            '\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm / count_norm, train_score_pos))
        if eval_loader is not None:
            logger.write('\t==> eval score: %.2f (%.2f)' %
                         (100 * eval_out['eval_score'], 100 * eval_out['bound']))
            logger.write('\teval y/n score: %.2f' %
                         (100 * eval_out['score_yesno']))
            logger.write('\teval other score: %.2f' %
                         (100 * eval_out['score_other']))
            logger.write('\teval number score: %.2f' %
                         (100 * eval_out['score_number']))

 
        if (eval_loader is not None and eval_out['eval_score'] > best_eval_score):
            best_eval_score = eval_out['eval_score']
            best_eval_epoch = epoch+1
            model_path = os.path.join(opt.output, 'best_model.pth')
            utils.save_model(model_path, model, epoch,
                             best_eval_score, best_eval_epoch, optim)
            logger.write('\n==========best==============\n')
        else:
            logger.write('\n\tbest_eval_epoch: %d\n\tbest_eval_score: %.2f\n' %
                         (best_eval_epoch, best_eval_score*100))

        if epoch < opt.pretrain_epoches: 
            model_path = os.path.join(opt.output, 'pretrain.pth')
            utils.save_model(model_path, model, epoch,
                             best_eval_score, best_eval_epoch, optim)

        # model_path = os.path.join(opt.output, 'epoch_%d.pth' % (epoch+1))
        # utils.save_model(model_path, model, epoch, best_eval_score, optim)


@ torch.no_grad()
def evaluate(model, dataloader, qid2type, opt):
    score = 0

    upper_bound = 0

    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for i, (v, box, q, a, qids, bias, _, Shuffling_q, Removal_q, positive_q, unbias_or_not) in tqdm(enumerate(dataloader), ncols=100, total=len(dataloader), desc="eval"):
        v = v.cuda()
        box = box.cuda()
        q = q.cuda()
        a = a.cuda()
        bias = bias.cuda()
        qids = qids.detach().cpu().int().numpy()

        out = model(q, v)

        pred = out['logits_pos']

        batch_score = compute_score_with_logits(
            pred, a.cuda()).cpu().numpy().sum(1)

        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum().item()

        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1

            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1

            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    eval_out = {}

    eval_out['eval_score'] = score
    eval_out['bound'] = upper_bound
    eval_out['score_yesno'] = score_yesno
    eval_out['score_other'] = score_other
    eval_out['score_number'] = score_number

    return eval_out
