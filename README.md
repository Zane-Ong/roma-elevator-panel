# RoMa 电梯面板矩形标定与匹配

使用 RoMa（CVPR 2024）把固定参考图上标定的按键矩形映射到新视角照片，输出与 YOLO 检测框类似的区域数据，便于接入 RGB-D 框内深度统计。提供本地网页、矩形拖框、固定模板、实时照片预览、匹配连线、配准叠加以及单按键掩膜。

本仓库包含项目基准照片和已标定的 11 个按钮：key_1–key_9、key_0、key_ok。更换电梯面板时，需要替换基准图和重新标定；它们不能直接用于布局不同的面板。

## 已保存的默认参数

统一配置文件：[rectangle_panel/settings.json](rectangle_panel/settings.json)。网页初始化、后端默认值和启动预热共用该文件。

| 参数 | 值 |
|---|---|
| 模型权重 | outdoor |
| 匹配置信度 | 0.5 |
| 参与配准的匹配点上限 | 10000 |
| 几何误差阈值 | 3 像素 |
| 基准图特征缓存 | 开启 |
| 粗／精细网络分辨率 | 560×560 / 864×864 |

网页中的参数可用于单次比较；要修改启动默认值，编辑 settings.json 并重启服务。已保存的按键框在 [rectangle_panel/template.json](rectangle_panel/template.json)。

## 安装与启动（Windows / Python 3.11）

RoMa 官方代码以 Git 子模块保存，固定到提交 `77f8d68803526dcddfd9b7a46bc76125bdc25f15`，请完整克隆：

```powershell
git clone --recurse-submodules https://github.com/Zane-Ong/roma-elevator-panel.git
cd roma-elevator-panel
python -m venv .venv
```

在独立环境安装 GPU 版 PyTorch，再安装网页与推理依赖。本机验证环境为 Python 3.11.9、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、RTX 4060 Laptop 8GB。其他驱动/显卡应按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择对应版本。

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps ./RoMa
.\.venv\Scripts\python.exe -X utf8 rectangle_server.py
```

启动后打开 **http://127.0.0.1:7861**，也可以双击“启动矩形标定版.bat”。首次运行会从官方地址下载所需权重到 models 文件夹；权重不上传 GitHub。首次加载与 GPU 预热较慢，状态栏显示完成后再运行。

部署主要在 Windows 验证，中文图像标注使用系统字体 `C:/Windows/Fonts/msyh.ttc`。迁移到 Linux 时需要替换字体路径、按环境安装 PyTorch；尚未验证完整 Linux 部署。

## 使用

1. 已标定时，标定区域默认收起；“展开标定区域”可编辑，拖框后点击保存。
2. 上传同一面板的实时照片，照片立即预览。
3. 点击匹配，查看特征匹配连线、两图配准叠加和实时按键框。
4. 下载结果包：mapping.json、current.png、matches.png、overlay.png、preview.png 和 11 张掩膜。

原版点标注页面通过 `app.py` 启动，在 **7860**。8GB 显存下不建议同时在两个页面加载完整 GPU 模型。

## 与深度算法接入

每个按键返回 `bbox_xyxy`（x1,y1,x2,y2，x2/y2 为切片结束位置）、投影四边形、中心和像素面积。框可以用于 `depth[y1:y2,x1:x2]`，过滤无效深度后计算原算法要求的统计量。掩膜与实时 RGB 原图同尺寸，255 表示区域内，可减少倾斜视角外接框包含的背景。

本工具只输出图像区域，未连接真实深度相机或机械臂。必须使用与本次实时 RGB 图对齐的深度，再接相机内参、手眼变换和控制接口。拟合残差不是按钮真实定位误差，落在图像内也不代表按钮无遮挡。H 假定面板近似平面。

详细字段与参数：[参数与接入说明](rectangle_panel/参数与接入说明.md)。

## 验证与性能

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest test_rectangle_geometry.py test_rectangle_template.py
```

测试覆盖四角投影、半开矩形框、掩膜像素数量、越界裁剪、无效标注拒绝、保存重载和原标定不被测试修改。

此前使用固定基准图的已知透视变化测试了完整 indoor 模型：基准特征缓存约 246MB，热运行网络推理中位耗时约 1.92→1.13 秒（1.69 倍）。这是同机小样本测试，不代表默认 outdoor 的速度或真实电梯精度；网络推理时间也不包含所有网页处理耗时。记录在 [performance.json](rectangle_panel/performance.json)。

## 来源与许可

官方论文：[RoMa: Robust Dense Feature Matching](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf)。官方代码：[Parskatt/RoMa](https://github.com/Parskatt/RoMa)。本工具使用完整 RoMa，没有替换为 Tiny RoMa。

RoMa 主体代码 MIT，DINOv2 部分 Apache-2.0，原许可证和声明保留在子模块内。本仓库是应用界面与区域映射适配，不是 RoMa 作者的官方仓库。
