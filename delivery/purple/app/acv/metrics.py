def rank_score(ranking, faulty_car, n=None):
    if faulty_car not in ranking: return 0.0
    n = len(ranking) if n is None else n
    return (n-ranking.index(faulty_car))/n

def case_result(file_id, ranking, faulty_car):
    rank = ranking.index(faulty_car)+1 if faulty_car in ranking else len(ranking)+1
    return {"file_id": file_id, "faulty_car": faulty_car, "ranking": ranking,
            "rank": rank, "score": rank_score(ranking, faulty_car), "top1": int(rank==1), "top2": int(rank<=2)}

def summary(rows):
    if not rows: raise ValueError("No cases to score")
    return {"cases": len(rows), "primary_metric": sum(r["score"] for r in rows)/len(rows),
            "top1_accuracy": sum(r["top1"] for r in rows)/len(rows),
            "top2_accuracy": sum(r["top2"] for r in rows)/len(rows),
            "worst_rank": max(r["rank"] for r in rows)}

def selection_key(rows, complexity=0):
    s = summary(rows)
    return (s["primary_metric"], -s["worst_rank"], s["top1_accuracy"], -complexity)
