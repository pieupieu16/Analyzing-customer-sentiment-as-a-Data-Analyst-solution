from typing import Dict, List

from .model import ASPECT_NAMES, ID2LABEL

LABELS = list(ID2LABEL.values())  # ["None", "Positive", "Negative", "Neutral"]


def build_summary(rows: List[Dict], skipped: int, truncated_from: int | None) -> Dict:
    by_aspect = {a: {l: 0 for l in LABELS} for a in ASPECT_NAMES}
    overall = {l: 0 for l in LABELS}
    by_rating_neg_count = {str(r): {a: 0 for a in ASPECT_NAMES} for r in range(1, 6)}
    by_rating_total = {str(r): 0 for r in range(1, 6)}

    for row in rows:
        r = str(int(row["rating"])) if 1 <= int(row["rating"]) <= 5 else None
        if r:
            by_rating_total[r] += 1
        for a in ASPECT_NAMES:
            label = row[a]
            by_aspect[a][label] += 1
            overall[label] += 1
            if label == "Negative" and r:
                by_rating_neg_count[r][a] += 1

    by_rating_aspect_neg = {}
    for r, asp_counts in by_rating_neg_count.items():
        total = by_rating_total[r]
        by_rating_aspect_neg[r] = {
            a: (asp_counts[a] / total if total else 0.0) for a in ASPECT_NAMES
        }

    return {
        "total": len(rows),
        "skipped": skipped,
        "truncated_from": truncated_from,
        "overall_sentiment": overall,
        "by_aspect": by_aspect,
        "by_rating_aspect_neg": by_rating_aspect_neg,
        "insights": [],
    }


def build_insights(summary: dict) -> list[str]:
    insights = []
    by_aspect = summary["by_aspect"]
    total = summary["total"]
    if total == 0:
        return ["Không có dữ liệu hợp lệ để phân tích."]

    MIN_MENTIONS = max(10, total // 20)  # at least 10 mentions, or 5% of total

    # Helper: per-aspect derived rates
    derived = {}
    for asp, counts in by_aspect.items():
        none_n = counts.get("None", 0)
        total_n = sum(counts.values())
        mentioned = total_n - none_n
        derived[asp] = {
            "mentioned": mentioned,
            "neg_rate_mentioned": (counts.get("Negative", 0) / mentioned) if mentioned else 0.0,
            "pos_rate_mentioned": (counts.get("Positive", 0) / mentioned) if mentioned else 0.0,
            "none_rate_total": (none_n / total_n) if total_n else 0.0,
        }

    # --- 1. Aspect with highest Negative-rate-among-mentions (with reliability floor) ---
    eligible_neg = {
        a: d["neg_rate_mentioned"]
        for a, d in derived.items()
        if d["mentioned"] >= MIN_MENTIONS
    }
    worst_asp = max(eligible_neg, key=eligible_neg.get) if eligible_neg else None
    if worst_asp and eligible_neg[worst_asp] > 0:
        insights.append(
            f"Khía cạnh có tỉ lệ **Negative** cao nhất (trong các review có nhắc): "
            f"**{worst_asp}** ({eligible_neg[worst_asp] * 100:.1f}%, "
            f"{derived[worst_asp]['mentioned']} lượt nhắc)."
        )

    # --- 2. Aspect with highest Positive-rate-among-mentions (different from #1) ---
    eligible_pos = {
        a: d["pos_rate_mentioned"]
        for a, d in derived.items()
        if d["mentioned"] >= MIN_MENTIONS
    }
    best_asp = None
    for asp, _ in sorted(eligible_pos.items(), key=lambda x: x[1], reverse=True):
        if asp != worst_asp:
            best_asp = asp
            break
    if best_asp and eligible_pos[best_asp] > 0.3:
        insights.append(
            f"Khía cạnh được khen nhiều nhất: **{best_asp}** "
            f"({eligible_pos[best_asp] * 100:.1f}% Positive trong số review có nhắc)."
        )

    # --- 3. Low ratings — complaint patterns ---
    by_rating = summary["by_rating_aspect_neg"]
    for r in ["1", "2"]:
        if r in by_rating and any(v > 0 for v in by_rating[r].values()):
            top_asp = max(by_rating[r], key=by_rating[r].get)
            rate = by_rating[r][top_asp]
            if rate > 0:
                insights.append(
                    f"Review **{r} sao** chủ yếu phàn nàn về **{top_asp}** "
                    f"({rate * 100:.0f}% Negative)."
                )

    # --- 4. Aspect that customers rarely mention (None > 30% of total) ---
    quiet_asp = max(derived, key=lambda a: derived[a]["none_rate_total"])
    if derived[quiet_asp]["none_rate_total"] > 0.3:
        insights.append(
            f"{derived[quiet_asp]['none_rate_total'] * 100:.0f}% review không đề cập "
            f"**{quiet_asp}** "
            f"— khía cạnh này ít gây chú ý."
        )

    return insights[:5]
