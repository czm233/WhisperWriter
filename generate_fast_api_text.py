import tkinter as tk
from tkinter import ttk
import re


def generate_django_code(model_name):
    # Convert snake_case to CamelCase
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))

    # Convert snake_case to kebab-case for URL basename
    basename = f"admin-{'-'.join(model_name.split('_'))}"

    # Generate model code
    model_code = f'''class {class_name}(Base):
    class Meta(Base.Meta):
        db_table = '{model_name}'
        verbose_name = ''
        verbose_name_plural = verbose_name'''

    # Generate serializer code
    serializer_code = f'''class {class_name}CreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.{class_name}
        exclude = ['created_at', 'updated_at', 'deleted_at']


class {class_name}Serializer(serializers.ModelSerializer):
    class Meta:
        model = models.{class_name}
        fields = '__all__\''''

    # Generate viewset code
    viewset_code = f'''class {class_name}ViewSet(viewsets.ModelViewSet):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAdminUser,)

    queryset = {class_name}.objects.all()
    serializer_class = {class_name}Serializer
    ordering_fields = ['id']

    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return {class_name}CreateSerializer
        return {class_name}Serializer'''

    # Convert snake_case to kebab-case for URL basename
    basename = re.sub('_', '-', model_name) # 使用正则表达式替换所有下划线为短横线
    url = f'''router.register(r'{model_name}', views.{''.join(word.capitalize() for word in model_name.split('_'))}ViewSet, basename='admin-{basename}')'''

    return {
        'model': model_code,
        'serializer': serializer_code,
        'viewset': viewset_code,
        'url': url,
    }


class CodeGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Django Code Generator")

        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 输入区域
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=0, column=0, pady=5)

        ttk.Label(input_frame, text="Model File Name:").pack(side=tk.LEFT)
        self.model_name = ttk.Entry(input_frame, width=40)
        self.model_name.pack(side=tk.LEFT, padx=5)

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=5)

        ttk.Button(button_frame, text="Generate", command=self.generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

        # 创建三个输出区域
        # Model输出
        ttk.Label(main_frame, text="Model:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.model_output = tk.Text(main_frame, wrap=tk.WORD, width=150, height=10)
        self.model_output.grid(row=3, column=0, pady=(0, 10))

        # ViewSet输出
        ttk.Label(main_frame, text="ViewSet:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.viewset_output = tk.Text(main_frame, wrap=tk.WORD, width=150, height=16)
        self.viewset_output.grid(row=5, column=0, pady=(0, 10))

        # Serializer输出
        ttk.Label(main_frame, text="Serializer:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.serializer_output = tk.Text(main_frame, wrap=tk.WORD, width=150, height=16)
        self.serializer_output.grid(row=7, column=0, pady=(0, 10))

        # url输出
        ttk.Label(main_frame, text="url:").grid(row=8, column=0, sticky=tk.W, pady=(10, 0))
        self.url_output = tk.Text(main_frame, wrap=tk.WORD, width=150, height=2)
        self.url_output.grid(row=9, column=0, pady=(0, 10))

        # 为每个文本框添加滚动条
        for text_widget in [self.model_output, self.serializer_output, self.viewset_output, self.url_output]:
            scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
            scrollbar.grid(row=text_widget.grid_info()['row'], column=1, sticky=(tk.N, tk.S))
            text_widget['yscrollcommand'] = scrollbar.set

    def generate(self):
        model_name = self.model_name.get().strip()
        if not model_name:
            return

        code = generate_django_code(model_name)

        # 清空并填充每个输出区域
        self.model_output.delete(1.0, tk.END)
        self.model_output.insert(tk.END, code['model'])

        self.viewset_output.delete(1.0, tk.END)
        self.viewset_output.insert(tk.END, code['viewset'])

        self.serializer_output.delete(1.0, tk.END)
        self.serializer_output.insert(tk.END, code['serializer'])

        self.url_output.delete(1.0, tk.END)
        self.url_output.insert(tk.END, code['url'])

    def clear(self):
        self.model_name.delete(0, tk.END)
        self.model_output.delete(1.0, tk.END)
        self.serializer_output.delete(1.0, tk.END)
        self.viewset_output.delete(1.0, tk.END)
        self.url_output.delete(1.0, tk.END)


if __name__ == '__main__':
    root = tk.Tk()
    app = CodeGeneratorGUI(root)
    root.mainloop()
