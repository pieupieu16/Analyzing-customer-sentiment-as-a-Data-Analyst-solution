"""
preprocess_dataset.py — Tiền xử lý Dataset_duy.csv để upload vào webapp.

Logic IDENTICAL với BTL_DataMining.ipynb (cell 14, 18, 30, 36):
  1. dropna(subset=['content'])                          ← cell 14
  2. drop_duplicates(subset=['review_id']) + drop 'purchased' ← cell 18 (handle_duplicates_and_missing)
  3. MIN_WORDS = 6, lọc dòng có len(split()) >= 6        ← cell 30
  4. filter_clothing_reviews() với blacklist + NFC normalize ← cell 36

Khác với notebook:
  - Notebook giữ 7 cột (rating, content, 5 cột nhãn). File Dataset_duy.csv KHÔNG có 5 cột nhãn,
    nên script chỉ giữ 2 cột: rating, content.
  - Thêm bước cast rating int 1-5 (notebook không cần vì rating đã là int).

Usage:
    python preprocess_dataset.py
    python preprocess_dataset.py --input Dataset_duy.csv --output Dataset_duy_clean.csv
    python preprocess_dataset.py --min-words 4
    python preprocess_dataset.py --no-filter   # bỏ qua filter blacklist
    python preprocess_dataset.py --keep-rating-content-only   # bỏ qua nếu file đã có nhãn
"""

import argparse
import os
import sys
import unicodedata
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_INPUT  = "Dataset_duy.csv"
DEFAULT_OUTPUT = "Dataset_duy_clean.csv"
DEFAULT_MIN_WORDS = 6

# IDENTICAL với cell 36 của notebook (mục __main__).
BLACKLIST = [
    'nhận xu', 'lấy xu', 'mang tính chất minh họa', 'mang tính chất minh hoạ',
    'tính chất nhận xu', 'hình ảnh mang', 'kiếm xu', 'để nhận', 'chỉ mang tính',
    'sách', 'quyển', 'trang sách', 'đọc', 'tác giả', 'nội dung sách',
    'truyện', 'tiểu thuyết', 'bookmark', 'fahasa', 'nhà sách', 'nhà xuất bản', 'bìa',
    'tác phẩm', 'nhân vật', 'hector malot', 'Câu chuyện', 'cậu chuyện'
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1258", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  [OK] Đọc thành công với encoding={enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Không đọc được {path} với bất kỳ encoding nào.")


def step(name: str, before: int, after: int) -> None:
    skipped = before - after
    pct = (skipped / before * 100) if before else 0
    print(f"  -> {name:<48} {before:>5} -> {after:>5}   (-{skipped}, {pct:.1f}%)")


# ---------------------------------------------------------------------------
# Pipeline — COPY 1-1 từ notebook
# ---------------------------------------------------------------------------

def handle_duplicates_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """COPY từ notebook cell 18."""
    # 1. Xóa các dòng trùng lặp dựa trên review_id
    df = df.drop_duplicates(subset=['review_id']).copy()
    # 2. Xóa bỏ cột 'purchased' do phần lớn là rỗng
    if 'purchased' in df.columns:
        df = df.drop(columns=['purchased'])
    return df


def filter_clothing_reviews(
    df: pd.DataFrame,
    text_col: str = 'content',
    remove_keywords: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """COPY 1-1 từ notebook cell 36."""
    if remove_keywords is None:
        remove_keywords = BLACKLIST

    df_filtered = df.copy()

    def normalize_text(text):
        if pd.isna(text):
            return ""
        # Chuẩn hóa Unicode về dạng NFC rồi mới lowercase
        return unicodedata.normalize('NFC', str(text)).lower()

    temp_col = 'temp_lower_text'
    df_filtered[temp_col] = df_filtered[text_col].apply(normalize_text)

    if remove_keywords:
        normalized_keywords = [unicodedata.normalize('NFC', kw).lower() for kw in remove_keywords]
        remove_pattern = '|'.join(normalized_keywords)
        mask_remove = df_filtered[temp_col].str.contains(remove_pattern, na=False, regex=True)
        df_removed = df_filtered[mask_remove]
        df_filtered = df_filtered[~mask_remove]
    else:
        df_removed = pd.DataFrame(columns=df_filtered.columns)

    df_filtered = df_filtered.drop(columns=[temp_col])
    if not df_removed.empty:
        df_removed = df_removed.drop(columns=[temp_col])

    return df_filtered, df_removed


def preprocess(df: pd.DataFrame,
               min_words: int = DEFAULT_MIN_WORDS,
               apply_keyword_filter: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_start = len(df)
    print(f"\n[Pipeline]  Dòng ban đầu: {n_start}")
    print(f"            Cột hiện có: {list(df.columns)}")

    if "content" not in df.columns:
        raise ValueError("CSV không có cột 'content' — bắt buộc cần cột này.")
    if "rating" not in df.columns:
        raise ValueError("CSV không có cột 'rating' — bắt buộc cần cột này.")

    # --- Cell 14: dropna content -----------------------------------------
    n_before = len(df)
    df = df.dropna(subset=['content']).copy()
    step("[cell 14] dropna(subset=['content'])", n_before, len(df))

    # --- Cell 18: drop dup review_id + drop 'purchased' ------------------
    n_before = len(df)
    if 'review_id' in df.columns:
        df = handle_duplicates_and_missing(df)
        step("[cell 18] drop_duplicates(review_id)", n_before, len(df))
    else:
        # Fallback nếu CSV không có review_id: drop trùng theo content
        df = df.drop_duplicates(subset=['content']).copy()
        step("[fallback] drop_duplicates(content)", n_before, len(df))

    # --- Cell 30: lọc câu < MIN_WORDS từ ---------------------------------
    n_before = len(df)
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
    df = df[df['word_count'] >= min_words].copy()
    df = df.drop(columns=['word_count'])
    step(f"[cell 30] giữ dòng >= {min_words} từ", n_before, len(df))

    # --- Cell 36: filter_clothing_reviews --------------------------------
    df_removed = pd.DataFrame()
    if apply_keyword_filter:
        n_before = len(df)
        df, df_removed = filter_clothing_reviews(df, text_col='content',
                                                 remove_keywords=BLACKLIST)
        step("[cell 36] filter blacklist (NFC + regex)", n_before, len(df))

    # --- Cast rating + giữ 2 cột (extension, không có trong notebook) ----
    n_before = len(df)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating']).copy()
    df['rating'] = df['rating'].astype(int).clip(1, 5)
    step("[extra] cast rating int 1-5", n_before, len(df))

    df = df[['rating', 'content']].reset_index(drop=True)

    print(f"\n[Kết quả]   {n_start} -> {len(df)} dòng  "
          f"(giữ lại {len(df)/n_start*100:.1f}%)")
    return df, df_removed


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n[Thống kê]")
    print("  Phân phối rating:")
    counts = df["rating"].value_counts().sort_index()
    mx = max(counts) if len(counts) else 1
    for r, c in counts.items():
        bar = "#" * int(c / mx * 30)
        print(f"    {r} sao  {c:>5}  {bar}")
    lengths = df["content"].str.len()
    print(f"  Độ dài content: min={lengths.min()}, "
          f"trung bình={lengths.mean():.0f}, max={lengths.max()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess Dataset_duy.csv giống notebook BTL_DataMining.ipynb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--no-filter", action="store_true",
                        help="Bỏ qua bước filter_clothing_reviews (cell 36).")
    parser.add_argument("--dump-removed", default=None,
                        help="Nếu set, ghi các dòng bị loại bỏ ở cell 36 vào file này.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    in_path  = os.path.join(here, args.input)  if not os.path.isabs(args.input)  else args.input
    out_path = os.path.join(here, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.exists(in_path):
        print(f"[LỖI] Không tìm thấy file: {in_path}")
        return 1

    print(f"[Input]   {in_path}")
    df = read_csv_robust(in_path)

    try:
        df_clean, df_removed = preprocess(
            df,
            min_words=args.min_words,
            apply_keyword_filter=not args.no_filter,
        )
    except ValueError as e:
        print(f"\n[LỖI] {e}")
        return 1

    print_stats(df_clean)

    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Output]  Đã ghi {len(df_clean)} dòng vào: {out_path}")
    print(f"          (UTF-8 BOM -> Excel mở không lỗi tiếng Việt)")

    if args.dump_removed and not df_removed.empty:
        dump_path = (os.path.join(here, args.dump_removed)
                     if not os.path.isabs(args.dump_removed) else args.dump_removed)
        df_removed.to_csv(dump_path, index=False, encoding="utf-8-sig")
        print(f"          + {len(df_removed)} dòng bị loại đã ghi vào: {dump_path}")

    print("\n[Xong] Upload file output này vào tab Batch của webapp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())