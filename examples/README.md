# Ví dụ xử lý dữ liệu

Các ví dụ cụ thể về cách xử lý dữ liệu thực tế cho hệ thống recommendation.

## Chạy ví dụ

```bash
# Ví dụ 1: Xử lý từ CSV files
python examples/process_data_example.py --example 1

# Ví dụ 2: Xử lý từ Database
python examples/process_data_example.py --example 2

# Ví dụ 3: Xử lý thủ công (mặc định)
python examples/process_data_example.py --example 3

# Ví dụ 4: Chỉ có events, tạo users/items
python examples/process_data_example.py --example 4
```

## Cấu trúc dữ liệu mẫu

### CSV Format

**products.csv:**
```csv
product_id,name,category,price,rating,sales
p_1,Laptop,electronics,999.99,4.5,1000
p_2,Phone,electronics,599.99,4.8,2000
```

**customers.csv:**
```csv
customer_id,age,gender,country,total_orders
u_1,25,M,VN,10
u_2,30,F,US,5
```

**interactions.csv:**
```csv
customer_id,product_id,action,timestamp
u_1,p_1,click,1700000000000
u_1,p_2,purchase,1700000100000
```

## Xem thêm

- [Hướng dẫn chi tiết](../../docs/DATA_PROCESSING_GUIDE.md)
- [Script xử lý tự động](../../scripts/process_real_data.py)

