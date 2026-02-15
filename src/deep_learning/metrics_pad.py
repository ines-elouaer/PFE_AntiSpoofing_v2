from collections import defaultdict

def aggregate_video_scores(frame_preds):
    
    bucket = defaultdict(list)
    labels = {}
    for p in frame_preds:
        vid = p["video_id"]
        bucket[vid].append(p["score_attack"])
        labels[vid] = p["label"]  
    video_scores = {}
    for vid, scores in bucket.items():
        video_scores[vid] = {"label": labels[vid], "score": sum(scores)/len(scores)}
    return video_scores

def compute_apcer_bpcer_acer(video_scores, threshold=0.5):
    
    n_attack = 0
    n_bona = 0
    attack_missed_as_real = 0   
    bona_missed_as_attack = 0  

    for vid, d in video_scores.items():
        y = d["label"]
        s = d["score"]
        pred_attack = 1 if s >= threshold else 0

        if y == 1:
            n_attack += 1
            if pred_attack == 0:
                attack_missed_as_real += 1
        else:
            n_bona += 1
            if pred_attack == 1:
                bona_missed_as_attack += 1

    apcer = (attack_missed_as_real / n_attack) if n_attack > 0 else 0.0
    bpcer = (bona_missed_as_attack / n_bona) if n_bona > 0 else 0.0
    acer = 0.5 * (apcer + bpcer)
    return apcer, bpcer, acer
