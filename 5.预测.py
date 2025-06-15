# 5.预测.py
import pandas as pd
import joblib
import sys
import os
import warnings
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QGridLayout,
    QGroupBox, QStatusBar, QMessageBox
)
from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QDoubleValidator
from PySide6.QtCore import Qt, QLocale

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

# 输入参数定义和合理范围
INPUT_PARAMS = [
    ("AQI_1天前", "前一天AQI指数", "", (0, 500)),
    ("PM2.5_1天前", "前一天PM2.5浓度", "μg/m³", (0, 300)),
    ("PM10_1天前", "前一天PM10浓度", "μg/m³", (0, 500)),
    ("So2_1天前", "前一天SO₂浓度", "μg/m³", (0, 100)),
    ("No2_1天前", "前一天NO₂浓度", "μg/m³", (0, 150)),
    ("O3_1天前", "前一天O₃浓度", "μg/m³", (0, 250)),
    ("Co_1天前", "前一天CO浓度", "mg/m³", (0, 5))
]


class AQIWidget(QWidget):
    def __init__(self, aqi_value):
        super().__init__()

        # 设置背景色和固定大小
        self.setFixedSize(100, 100)
        self.aqi_value = aqi_value
        self.setup_ui()

    def setup_ui(self):
        # 创建垂直布局
        layout = QVBoxLayout(self)

        # 添加标签显示AQI值
        aqi_label = QLabel(str(int(self.aqi_value)))
        aqi_font = QFont("Arial", 24, QFont.Bold)
        aqi_label.setFont(aqi_font)
        aqi_label.setAlignment(Qt.AlignCenter)

        # 添加标签显示空气质量等级
        level_label = QLabel("")
        level_font = QFont("Arial", 12, QFont.Bold)
        level_label.setFont(level_font)
        level_label.setAlignment(Qt.AlignCenter)

        # 设置背景颜色基于AQI值
        palette = self.palette()

        if self.aqi_value <= 50:
            level_label.setText("优")
            palette.setColor(QPalette.Window, QColor(0, 228, 0))  # 绿色
        elif self.aqi_value <= 100:
            level_label.setText("良")
            palette.setColor(QPalette.Window, QColor(255, 255, 0))  # 黄色
        elif self.aqi_value <= 150:
            level_label.setText("轻度")
            palette.setColor(QPalette.Window, QColor(255, 126, 0))  # 橙色
        elif self.aqi_value <= 200:
            level_label.setText("中度")
            palette.setColor(QPalette.Window, QColor(255, 0, 0))  # 红色
        elif self.aqi_value <= 300:
            level_label.setText("重度")
            palette.setColor(QPalette.Window, QColor(153, 0, 76))  # 紫色
        else:
            level_label.setText("严重")
            palette.setColor(QPalette.Window, QColor(126, 0, 35))  # 褐红色

        self.setAutoFillBackground(True)
        self.setPalette(palette)

        # 添加到布局
        layout.addWidget(aqi_label)
        layout.addWidget(level_label)


class AirQualityPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.models = self.load_models()
        self.init_ui()
        # 添加白底黑字主题
        self.apply_white_theme()

    def apply_white_theme(self):
        """应用白底黑字主题"""
        # 设置窗口背景为白色
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
                font-weight: bold;
                color: black;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: black;
            }
            QLabel {
                background-color: transparent;
                color: black;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 5px;
                color: black;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 5px;
                color: black;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 5px 10px;
                color: black;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                color: black;
            }
            QStatusBar {
                background-color: #f5f5f5;
                color: black;
                border-top: 1px solid #d0d0d0;
            }
        """)

    def load_models(self):
        """加载所有可用模型"""
        models = {}
        models_dir = 'models'

        # 检查models目录是否存在
        if not os.path.exists(models_dir):
            print(f"警告: 模型目录 '{models_dir}' 不存在")
            return models

        print(f"从目录 '{models_dir}' 加载模型...")

        # 支持的模型文件列表
        model_files = [
            '随机森林_model.pkl',
            '梯度提升_model.pkl',
            '线性回归_model.pkl',
            '支持向量机_model.pkl',
            'K近邻_model.pkl'
        ]

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace('_model.pkl', '')

            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"警告: 模型文件 '{model_path}' 不存在")
                continue

            try:
                print(f"加载模型: {model_name}")
                model = joblib.load(model_path)
                models[model_name] = model
                print(f"✅ 成功加载模型: {model_name}")
            except Exception as e:
                print(f"❌ 加载模型 {model_name} 时出错: {str(e)}")

        # 检查是否加载了任何模型
        if not models:
            print("警告: 没有加载任何模型!")
        else:
            print(f"成功加载 {len(models)} 个模型")

        return models

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("空气质量预测系统")
        self.setMinimumSize(800, 600)

        # 设置主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 标题
        title_label = QLabel("空气质量预测系统")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 2. 控制区域
        control_group = QGroupBox("预测参数")
        control_layout = QGridLayout(control_group)
        control_layout.setColumnStretch(2, 1)  # 添加弹性空间

        # 创建输入字段
        self.input_fields = {}

        # 创建验证器，允许输入小数
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        validator = QDoubleValidator()
        validator.setLocale(locale)
        validator.setNotation(QDoubleValidator.StandardNotation)

        for i, (param_name, description, unit, value_range) in enumerate(INPUT_PARAMS):
            row = i
            col = 0

            # 添加标签
            label_text = f"{param_name}\n({description})"
            label = QLabel(label_text)
            label.setToolTip(f"合理范围: {value_range[0]} ~ {value_range[1]}")
            label_font = QFont("Arial", 10)
            label.setFont(label_font)

            # 添加输入框
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"{value_range[0]}-{value_range[1]}")
            line_edit.setMinimumWidth(150)
            line_edit.setValidator(validator)  # 设置验证器只允许数字输入
            self.input_fields[param_name] = line_edit

            # 添加单位标签
            unit_label = QLabel(unit)
            unit_label.setFont(label_font)

            # 添加到布局
            control_layout.addWidget(label, row, col)
            control_layout.addWidget(line_edit, row, col + 1)
            control_layout.addWidget(unit_label, row, col + 2)

        main_layout.addWidget(control_group)

        # 3. 模型选择和按钮区域
        options_layout = QHBoxLayout()

        # 模型选择
        model_label = QLabel("预测模型:")
        label_font = QFont("Arial", 11)
        model_label.setFont(label_font)

        self.model_combo = QComboBox()

        # 只添加实际加载的模型
        if self.models:
            self.model_combo.addItems(list(self.models.keys()))
            self.model_combo.setCurrentIndex(0)
        else:
            # 如果没有模型，添加一个警告项
            self.model_combo.addItem("⚠️ 没有可用的模型")

        self.model_combo.setMinimumWidth(150)

        # 按钮
        self.predict_button = QPushButton("预测")
        self.predict_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        self.predict_button.setMinimumSize(100, 40)
        self.predict_button.clicked.connect(self.predict)

        self.clear_button = QPushButton("清空")
        self.clear_button.setFont(QFont("Arial", 11))
        self.clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px 16px;")
        self.clear_button.setMinimumSize(100, 40)
        self.clear_button.clicked.connect(self.clear_fields)

        self.info_button = QPushButton("参数说明")
        self.info_button.setFont(QFont("Arial", 11))
        self.info_button.setStyleSheet("background-color: #2196F3; color: white; padding: 8px 16px;")
        self.info_button.setMinimumSize(100, 40)
        self.info_button.clicked.connect(self.show_param_info)

        options_layout.addWidget(model_label)
        options_layout.addWidget(self.model_combo)
        options_layout.addStretch(1)
        options_layout.addWidget(self.info_button)
        options_layout.addWidget(self.clear_button)
        options_layout.addWidget(self.predict_button)

        main_layout.addLayout(options_layout)

        # 4. 结果显示区域
        result_group = QGroupBox("预测结果")
        result_layout = QVBoxLayout(result_group)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Arial", 12))

        # 添加AQI可视化组件
        self.aqi_widget = QWidget()
        aqi_layout = QHBoxLayout(self.aqi_widget)
        aqi_layout.setAlignment(Qt.AlignCenter)
        self.aqi_layout = aqi_layout

        # 初始显示AQI为0
        self.update_aqi_display(0)

        result_layout.addWidget(self.result_display, 60)
        result_layout.addWidget(self.aqi_widget, 40)
        main_layout.addWidget(result_group)

        # 5. 设置状态栏
        self.status_bar = QStatusBar()
        if self.models:
            self.status_bar.showMessage("准备预测 - 输入前一天的空气质量数据")
        else:
            self.status_bar.showMessage("警告: 没有可用的模型 - 请先运行模型训练脚本")
        self.setStatusBar(self.status_bar)

        # 设置中心部件
        self.setCentralWidget(central_widget)

    def show_param_info(self):
        """显示参数说明对话框 - 优化界面大小和布局"""
        # 创建对话框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("空气质量预测参数说明")
        msg_box.setTextFormat(Qt.RichText)

        # 构建更详细的信息文本
        info_text = """
        <div style='font-family: Arial; font-size: 14px;'>
            <h2 style='color: #2c4e87; text-align: center; margin-bottom: 20px;'>
                空气质量预测参数说明
            </h2>
            <table border='1' cellspacing='0' cellpadding='10' 
                   style='border-collapse: collapse; width: 100%; margin: 15px 0;'>
                <tr style='background-color: #4a7caf; color: black;'>
                    <th  width='20%'>参数名称</th>
                    <th width='30%'>参数说明</th>
                    <th width='15%'>单位</th>
                    <th width='35%'>合理范围</th>
                </tr>
        """

        # 添加参数行
        for name, description, unit, (min_val, max_val) in INPUT_PARAMS:
            info_text += f"""
                <tr style='background-color: #f0f5ff;'>
                    <td  style='color: #000000; margin-top: 0;' style='text-align: center;'>{name}</td>
                    <td  style='color: #000000; margin-top: 0;' style='text-align: center;'>{description}</td>
                    <td  style='color: #000000; margin-top: 0;' style='text-align: center;'>{unit if unit else '-'}</td>
                    <td  style='color: #000000; margin-top: 0;' style='text-align: center;'>{min_val} - {max_val}</td>
                </tr>
            """

        info_text += """
            </table>#2c4e87

            <div style='margin: 20px 0; border-left: 5px solid #4a7caf; padding: 10px 15px; background-color: #f8faff;'>
                <h3 style='color: #000000; margin-top: 0;'>使用说明</h3>
                <p style='color: #000000; margin-top: 0;' >1. 所有参数均基于前一天空气质量测量值</p>
                <p style='color: #000000; margin-top: 0;'>2. 请输入数值时，请参考"合理范围"列</p>
                <p style='color: #000000; margin-top: 0;'>3. 参数值应介于最小值和最大值之间</p>
                <p style='color: #000000; margin-top: 0;'>4. 参数值超出范围将出现警告提示</p>
                <p style='color: #000000; margin-top: 0;'>5. 所有字段必须填写才能进行预测</p>
            </div>

            <div style='text-align: center; margin-top: 20px; color: #666;'>
                <p>空气质量预测系统 &copy; 2025</p>
            </div>
        </div>
        """

        msg_box.setText(info_text)

        # 设置对话框大小并固定尺寸
        msg_box.setMinimumSize(800, 600)
        msg_box.setMaximumSize(900, 700)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #f0f7ff;
            }
            QLabel#qt_msgbox_label {
                min-height: 500px;
                min-width: 750px;
            }
        """)
        # 移除标准按钮栏（因为我们只需要OK按钮）
        msg_box.setStandardButtons(QMessageBox.Ok)

        # 自定义OK按钮
        ok_button = msg_box.button(QMessageBox.Ok)
        ok_button.setText("确认")
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4a7caf;
                color: black;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 25px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #356294;
            }
        """)

        # 显示对话框
        msg_box.exec_()

    def clear_fields(self):
        """清空输入字段"""
        for field in self.input_fields.values():
            field.clear()
        self.result_display.clear()
        self.update_aqi_display(0)  # 重置AQI显示
        self.status_bar.showMessage("所有输入已清空")

    def update_aqi_display(self, aqi_value):
        """更新AQI显示组件"""
        # 清除现有AQI显示
        while self.aqi_layout.count():
            child = self.aqi_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # 创建容器
        container = QWidget()
        layout = QHBoxLayout(container)

        if aqi_value > 0:
            # 添加AQI数值显示
            aqi_display = AQIWidget(aqi_value)
            layout.addWidget(aqi_display)

            # 添加空气质量描述
            desc_widget = QWidget()
            desc_layout = QVBoxLayout(desc_widget)

            # 添加AQI数值
            aqi_value_label = QLabel(f"空气质量指数: {aqi_value:.0f}")
            aqi_value_label.setFont(QFont("Arial", 14, QFont.Bold))
            aqi_value_label.setAlignment(Qt.AlignCenter)

            # 添加等级和健康影响
            level, description = self.get_air_quality_description(aqi_value)
            level_label = QLabel(f"空气质量等级: {level}")
            level_label.setFont(QFont("Arial", 12))
            level_label.setAlignment(Qt.AlignCenter)

            desc_label = QLabel(f"健康影响: {description}")
            desc_label.setFont(QFont("Arial", 10))
            desc_label.setAlignment(Qt.AlignCenter)
            desc_label.setWordWrap(True)

            desc_layout.addWidget(aqi_value_label)
            desc_layout.addWidget(level_label)
            desc_layout.addWidget(desc_label)

            layout.addWidget(desc_widget)

            container.setVisible(True)
        else:
            # 当AQI为0时，显示提示信息
            if self.models:
                prompt_label = QLabel("请填写参数并点击'预测'按钮")
            else:
                prompt_label = QLabel("⚠️ 没有可用的模型 - 请先运行模型训练脚本")

            prompt_label.setFont(QFont("Arial", 14, QFont.Bold))
            prompt_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(prompt_label)
            container.setStyleSheet("background-color: #F0F0F0;")

        self.aqi_layout.addWidget(container)

    def get_air_quality_description(self, aqi):
        """获取空气质量描述信息"""
        if aqi <= 50:
            return "优", "空气质量令人满意，基本无空气污染，各类人群可正常活动。"
        elif aqi <= 100:
            return "良", "空气质量可接受，某些污染物可能对极少数敏感人群健康有较弱影响，建议极少数敏感人群减少户外活动。"
        elif aqi <= 150:
            return "轻度污染", "易感人群症状有轻度加剧，健康人群出现刺激症状。建议儿童、老年人及心脏病、呼吸系统疾病患者减少长时间、高强度的户外锻炼。"
        elif aqi <= 200:
            return "中度污染", "进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响。建议儿童、老年人及心脏病、呼吸系统疾病患者避免长时间、高强度的户外锻炼，一般人群适量减少户外运动。"
        elif aqi <= 300:
            return "重度污染", "心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群中普遍出现症状。建议儿童、老年人和病人应停留在室内停止户外运动，一般人群减少户外活动。"
        else:
            return "严重污染", "健康人群运动耐受力降低，有明显强烈症状，提前出现某些疾病。建议儿童、老年人和病人应当留在室内避免体力消耗，一般人群避免户外活动。"

    def validate_input(self, name, value):
        """验证输入值是否在合理范围内"""
        param_info = next(p for p in INPUT_PARAMS if p[0] == name)
        min_val, max_val = param_info[3]

        if value < min_val:
            QMessageBox.warning(
                self, "输入值过低",
                f"{name} 的值不能低于 {min_val}\n您输入的是: {value}"
            )
            return False

        if value > max_val:
            QMessageBox.warning(
                self, "输入值过高",
                f"{name} 的值不能超过 {max_val}\n您输入的是: {value}"
            )
            return False

        return True

    def predict(self):
        """执行预测功能"""
        # 如果没有可用的模型，显示警告
        if not self.models:
            QMessageBox.warning(
                self,
                "没有可用的模型",
                "没有找到任何可用的预测模型。\n\n请确保您已经运行了模型训练脚本，并且模型文件保存在'models'目录中。"
            )
            self.status_bar.showMessage("错误: 没有可用的模型")
            return

        self.status_bar.showMessage("正在验证输入...")

        # 获取输入值
        input_data = {}
        errors = []

        try:
            for name, field in self.input_fields.items():
                value_text = field.text().strip()
                if not value_text:
                    errors.append(f"未输入 {name} 的值")
                    continue

                try:
                    value = float(value_text)
                except ValueError:
                    errors.append(f"{name} 必须为有效数值")
                    continue

                # 验证输入范围
                if not self.validate_input(name, value):
                    errors.append(f"{name} 的值超出合理范围")

                # 存储值
                input_data[name] = value

            if errors:
                error_msg = "发现以下错误:\n- " + "\n- ".join(errors)
                self.result_display.setText(error_msg)
                self.status_bar.showMessage("输入验证失败")
                return

        except Exception as e:
            self.result_display.setText(f"错误: {str(e)}")
            self.status_bar.showMessage(f"输入验证错误: {str(e)}")
            return

        # 选择模型
        model_name = self.model_combo.currentText()
        if model_name not in self.models:
            self.result_display.setText(f"错误: 未找到模型 {model_name}")
            self.status_bar.showMessage(f"未找到模型 {model_name}")
            return

        self.status_bar.showMessage("正在执行预测...")

        # 创建DataFrame - 确保字段顺序正确
        field_names = [name for name, _, _, _ in INPUT_PARAMS]
        input_values = [input_data[name] for name in field_names]
        input_df = pd.DataFrame([input_values], columns=field_names)

        # 预测
        try:
            prediction = self.models[model_name].predict(input_df)
            aqi_value = prediction[0]

            # 获取空气质量描述
            level, description = self.get_air_quality_description(aqi_value)

            # 显示结果
            result_text = f"<b>预测模型</b>: {model_name}<br>"
            result_text += f"<b>预测AQI指数</b>: {aqi_value:.0f}<br>"
            result_text += f"<b>空气质量等级</b>: {level}<br><br>"
            result_text += f"<b>健康影响</b>: {description}<br><br>"
            result_text += "<b>输入参数</b>:<br>"

            for param in INPUT_PARAMS:
                name, desc, unit, _ = param
                value = input_data[name]
                unit_text = f" {unit}" if unit else ""
                result_text += f"- {name}: {value}{unit_text} ({desc})<br>"

            self.result_display.setText(result_text)

            # 更新AQI可视化
            self.update_aqi_display(aqi_value)

            self.status_bar.showMessage(f"预测完成 - AQI: {aqi_value:.0f} ({level})")
        except Exception as e:
            error_msg = f"<b>预测错误</b>: {str(e)}"
            error_msg += "<br><br>可能原因:<br>"
            error_msg += "- 模型文件已损坏<br>"
            error_msg += "- 输入数据格式不正确<br>"
            error_msg += "- 程序依赖库版本冲突"

            self.result_display.setText(error_msg)
            self.status_bar.showMessage(f"预测错误: {str(e)}")
            print(f"预测错误: {str(e)}")


if __name__ == "__main__":
    # 创建应用
    app = QApplication(sys.argv)

    # 移除原来的样式设置
    # app.setStyle("Fusion")  # 注释掉这一行

    # 创建主窗口
    window = AirQualityPredictionApp()
    window.show()

    # 运行应用
    sys.exit(app.exec())