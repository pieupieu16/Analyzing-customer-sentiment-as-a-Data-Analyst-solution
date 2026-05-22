import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import time

# Fix Unicode error on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load biến môi trường từ file .env (nếu có)
load_dotenv()

# Khởi tạo client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Định nghĩa cấu trúc đầu ra bằng Pydantic cho Structured Outputs
class ReviewLabels(BaseModel):
    nhan_chat_lieu: Optional[Literal["Positive", "Negative", "Neutral", "None"]] = Field(description="Nhãn cảm xúc cho khía cạnh chất liệu. Nội dung nhắc đến loại vải, cảm giác bề mặt, độ mỏng/dày, mềm/cứng, nóng/mát, co giãn, thấm hút mồ hôi. Trả về None nếu không đề cập.")
    nhan_kich_co_form: Optional[Literal["Positive", "Negative", "Neutral", "None"]] = Field(description="Nhãn cảm xúc cho khía cạnh kích cỡ và form. Nội dung nhắc đến size, chiều cao cân nặng, form rộng/chật/ngắn/dài, cảm giác vừa vặn, tôn dáng. Trả về None nếu không đề cập.")
    nhan_thiet_ke: Optional[Literal["Positive", "Negative", "Neutral", "None"]] = Field(description="Nhãn cảm xúc cho khía cạnh thiết kế và thẩm mỹ. Nội dung nhắc đến vẻ đẹp, màu sắc, hoa văn, trẻ trung, sang trọng, già dặn. Trả về None nếu không đề cập.")
    nhan_gia_cong: Optional[Literal["Positive", "Negative", "Neutral", "None"]] = Field(description="Nhãn cảm xúc cho khía cạnh kỹ thuật gia công. Nội dung nhắc đến đường kim mũi chỉ, chỉ thừa, vắt sổ, dây kéo, khuy nút. Trả về None nếu không đề cập.")
    nhan_gia_tri_thuc_te: Optional[Literal["Positive", "Negative", "Neutral", "None"]] = Field(description="Nhãn cảm xúc cho khía cạnh giá trị thực tế. Nội dung nhắc đến giá tiền có đáng không, hàng có giống hình/mô tả không, uy tín shop. Trả về None nếu không đề cập.")

# System prompt
SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích cảm xúc đánh giá sản phẩm thời trang. 
Nhiệm vụ của bạn là đọc thông tin đánh giá (bao gồm Tiêu đề, Nội dung, Số sao) và gán nhãn cảm xúc cho 5 khía cạnh cụ thể.
Mỗi khía cạnh phải được gán 1 trong 4 nhãn: "Positive", "Negative", "Neutral", hoặc "None".

I. QUY TẮC PHÂN LOẠI CẢM XÚC CHUNG:
- Positive (Tích cực): Khách hàng hài lòng, khen ngợi, thích thú về khía cạnh đó.
- Negative (Tiêu cực): Khách hàng chê bai, phàn nàn, thất vọng, giận dữ về khía cạnh đó.
- Neutral (Trung tính): Khách nêu ý kiến trần thuật, hoặc kiểu "chấp nhận được", "tiền nào của nấy", nhượng bộ, "tạm ổn", "cũng được".
- None (Không đề cập): Hoàn toàn không nhắc tới khía cạnh này. Hoặc nếu khách hàng chửi/phàn nàn đơn vị vận chuyển/shipper thì KHÔNG LIÊN QUAN chất lượng quần áo, gán None cho cả 5 cột.

II. ĐỊNH NGHĨA 5 KHÍA CẠNH:
1. nhan_chat_lieu (Chất liệu): Nhắc đến tên loại vải (cotton, poly, lanh), cảm giác bề mặt (thoáng, mát, nóng, nực, mỏng, dày, thô, ráp, mềm, cứng), hoặc độ co giãn, thấm hút mồ hôi.
2. nhan_kich_co_form (Kích cỡ & Trải nghiệm mặc): Nhắc đến chỉ số (size, chiều cao, cân nặng, số đo), form dáng rộng/chật/ngắn/dài, cảm giác ướm thử (kích, thùng thình, tôn dáng, vừa in).
3. nhan_thiet_ke (Thiết kế & Thẩm mỹ): Nhắc đến vẻ đẹp bề ngoài, sắc độ màu (sáng, xỉn màu, đậm, nhạt), hoa văn họa tiết, độ xòe, tính thời trang (sang trọng, già dặn, trẻ trung, quê mùa).
4. nhan_gia_cong (Kỹ thuật gia công & Phụ kiện): Yếu tố cơ học gắn kết (đường kim, mũi chỉ, chỉ thừa, vắt sổ, nếp gấp, may ẩu) hoặc phụ kiện (dây đai, dây thắt lưng, khóa kéo/zip, nút cài/khuy áo).
5. nhan_gia_tri_thuc_te (Giá trị & Độ trung thực): Đánh giá về số tiền bỏ ra (đáng tiền, rẻ, đắt), so sánh hàng thật với hình quảng cáo (giống hình / khác hình mạng / lừa đảo), hoặc sự uy tín của gian hàng.

III. XỬ LÝ NGOẠI LỆ:
- Một câu có nhiều khía cạnh chéo nhau: Gán độc lập. VD: "Áo đẹp y hình (gia_tri_thuc_te: Positive, thiet_ke: Positive) mà vải dởm quá (chat_lieu: Negative)".
- Một khía cạnh vừa khen vừa chê: Gán Neutral. VD: "Vải mỏng nhưng mặc mát" -> chat_lieu: Neutral.

Lưu ý: Bạn phải trả kết quả đúng định dạng JSON được yêu cầu. Các khía cạnh không có thông tin phải trả về "None".
"""
def get_labels_from_openai(row, retries=5):
    """Hàm gọi OpenAI API để gán nhãn cho 1 dòng dữ liệu có xử lý Rate Limit."""
    title = str(row.get('title', ''))
    content = str(row.get('content', ''))
    rating = str(row.get('rating', ''))
    
    user_prompt = f"Tiêu đề: {title}\nĐánh giá: {content}\nSố sao: {rating} sao"
    
    for attempt in range(retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ReviewLabels,
                temperature=0.1,
                max_tokens=256
            )
            
            parsed_result = response.choices[0].message.parsed
            return {
                'nhan_chat_lieu': parsed_result.nhan_chat_lieu if parsed_result.nhan_chat_lieu != "None" else None,
                'nhan_kich_co_form': parsed_result.nhan_kich_co_form if parsed_result.nhan_kich_co_form != "None" else None,
                'nhan_thiet_ke': parsed_result.nhan_thiet_ke if parsed_result.nhan_thiet_ke != "None" else None,
                'nhan_gia_cong': parsed_result.nhan_gia_cong if parsed_result.nhan_gia_cong != "None" else None,
                'nhan_gia_tri_thuc_te': parsed_result.nhan_gia_tri_thuc_te if parsed_result.nhan_gia_tri_thuc_te != "None" else None,
                'success': True
            }
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                wait_time = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
                print(f"Gặp lỗi Rate Limit ở review_id {row.get('review_id')}. Thử lại sau {wait_time}s... (Lần {attempt+1}/{retries})")
                time.sleep(wait_time)
            else:
                print(f"Lỗi khi xử lý review_id {row.get('review_id')}: {e}")
                return {'success': False, 'error': str(e)}
                
    return {'success': False, 'error': 'Max retries exceeded for Rate Limit'}

def main():
    input_file = "Dataset_quan.csv"
    output_file = "Dataset_quan_relabeled.csv"
    
    print(f"Đang đọc dữ liệu từ {input_file}...")
    df = pd.read_csv(input_file)
    
    # Kiểm tra xem file output đã tồn tại chưa để resume
    processed_review_ids = set()
    if os.path.exists(output_file):
        df_out = pd.read_csv(output_file)
        if 'review_id' in df_out.columns:
            processed_review_ids = set(df_out['review_id'].astype(str))
            print(f"Đã tìm thấy {len(processed_review_ids)} dòng đã xử lý. Tiếp tục từ phần còn lại...")
            # Khởi tạo file ở mode append không cần ghi lại header
            write_mode = 'a'
            write_header = False
        else:
            write_mode = 'w'
            write_header = True
    else:
        write_mode = 'w'
        write_header = True

    # Lọc những row chưa xử lý
    df_to_process = df[~df['review_id'].astype(str).isin(processed_review_ids)].copy()
    
    if len(df_to_process) == 0:
        print("Tất cả các dòng đã được xử lý xong!")
        return
        
    print(f"Còn {len(df_to_process)} dòng cần xử lý.")
    
    # Sử dụng ThreadPoolExecutor để gọi API song song
    max_workers = 3 # Đã giảm xuống từ 10 xuống 3 để tránh lỗi Rate Limit
    
    # Mở file CSV để ghi kết quả liên tục
    with open(output_file, mode=write_mode, encoding='utf-8-sig', newline='') as f:
        # Nếu là file mới thì ghi Header trước
        if write_header:
            header = list(df.columns)
            pd.DataFrame(columns=header).to_csv(f, index=False)
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Gửi task vào queue
            future_to_row = {executor.submit(get_labels_from_openai, row): row for index, row in df_to_process.iterrows()}
            
            # Khởi tạo progress bar
            with tqdm(total=len(df_to_process), desc="Relabeling") as pbar:
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    result = future.result()
                    
                    if result['success']:
                        # Cập nhật nhãn mới vào dòng hiện tại
                        row['nhan_chat_lieu'] = result['nhan_chat_lieu']
                        row['nhan_kich_co_form'] = result['nhan_kich_co_form']
                        row['nhan_thiet_ke'] = result['nhan_thiet_ke']
                        row['nhan_gia_cong'] = result['nhan_gia_cong']
                        row['nhan_gia_tri_thuc_te'] = result['nhan_gia_tri_thuc_te']
                        
                        # Chuyển series row thành dataframe 1 dòng và ghi thẳng vào file
                        pd.DataFrame([row]).to_csv(f, header=False, index=False)
                        f.flush() # Bắt buộc flush để đảm bảo ghi xuống ổ cứng
                    
                    pbar.update(1)

    print(f"\\nĐã xử lý xong. Dữ liệu được lưu tại {output_file}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("LỖI: Chưa cấu hình OPENAI_API_KEY. Vui lòng thêm vào file .env hoặc cấu hình biến môi trường.")
    else:
        main()
