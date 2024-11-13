import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL.Image import Resampling

# 导入外部功能模块
from pest_detection import detect_pests
from fruit_classfy import classify_fruit

# 创建主窗口
root = tk.Tk()
root.title("Fruit and Pest Detection")

# 设置窗口大小和背景颜色
root.geometry("800x600")
root.configure(bg='light blue')  # 添加背景颜色

# 加载和显示图片
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((400, 400), Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel_image.config(image=img_tk)
        panel_image.image = img_tk
        panel_image.pack()

        pest_result = detect_pests(file_path)
        fruit_result = classify_fruit(file_path)
        result_text = f"{pest_result}\n{fruit_result}"
        label_result.config(text=result_text)

# 使用 Frame 组织布局
frame = tk.Frame(root, bg='white', bd=2, relief=tk.RAISED)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# 图片面板
panel_image = tk.Label(frame)
panel_image.pack(pady=10)

# 结果标签
label_result = tk.Label(frame, text="Results will be shown here.", font=('Arial', 14), bg='white')
label_result.pack(pady=10)

# 按钮加载图片
btn_load = tk.Button(frame, text="Load Image", command=load_image)
btn_load.pack(pady=10)

# 运行主循环
root.mainloop()
