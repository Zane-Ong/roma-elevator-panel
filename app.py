from runtime_config import ROOT
import gradio as gr
from matcher_demo import DEVICE, label_points, parse_points, run_match


def add_button(reference, name, coordinates, event: gr.SelectData):
    if reference is None:
        return coordinates, None
    count = len([line for line in coordinates.splitlines() if line.strip()])
    name = name.strip() or f"按钮{count + 1}"
    name = name.replace(",", "_").replace("，", "_").replace("\n", " ")
    x, y = event.index
    coordinates = (coordinates.rstrip() + f"\n{name},{x},{y}").strip()
    return coordinates, label_points(reference.convert("RGB"), parse_points(coordinates, reference.size), "reference_xy")


def match(reference, current, variant, coordinates, confidence):
    try:
        return run_match(reference, current, variant, coordinates, confidence)
    except Exception as error:
        raise gr.Error(str(error)) from error


with gr.Blocks(title="RoMa 电梯面板匹配", analytics_enabled=False) as demo:
    gr.Markdown("# RoMa 电梯面板匹配\n上传同一面板的两张照片，比较不同视角下的匹配效果。建议保留数字、标识和面板边缘。")
    with gr.Row():
        reference = gr.Image(type="pil", sources=["upload"], label="① 参考照片（可点击按钮中心）", height=380)
        current = gr.Image(type="pil", sources=["upload"], label="② 实时照片", height=380)
    with gr.Row():
        variant = gr.Radio(["outdoor", "indoor"], value="outdoor", label="权重（可以分别比较）")
        confidence = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="匹配置信度门槛")
    with gr.Accordion("可选：标注按钮并查看映射位置", open=False):
        gr.Markdown("填写按钮名称，再点击左上参考照片中的中心；重复操作可添加多个按钮。只看匹配效果时可跳过。")
        name = gr.Textbox(label="下一次点击的按钮名称", placeholder="例如 key_1 或 确认键")
        coordinates = gr.Textbox(label="参考按钮记录（可编辑：名称,x,y）", value="", lines=4)
        clear = gr.Button("清空按钮标注")
        annotation = gr.Image(label="参考图标注预览", interactive=False, format="png")
    submit = gr.Button("③ 开始匹配", variant="primary")
    status = gr.Textbox(label="匹配结果", interactive=False, lines=3)
    with gr.Row():
        connections = gr.Image(label="几何一致的匹配连线", interactive=False, format="png")
        overlay = gr.Image(label="配准叠加（观察数字和边缘是否重合）", interactive=False, format="png")
    buttons = gr.Image(label="实时图上的按钮映射位置", interactive=False, format="png")
    download = gr.File(label="下载结果图片与坐标")
    gr.Markdown(f"运行设备：{'NVIDIA GPU' if DEVICE == 'cuda' else 'CPU'}。完整 RoMa 默认分辨率。红点是标注的映射位置，需要检查按钮身份和遮挡。")
    reference.select(add_button, [reference, name, coordinates], [coordinates, annotation], queue=False)
    reference.upload(lambda: ("", None), outputs=[coordinates, annotation], queue=False)
    reference.clear(lambda: ("", None), outputs=[coordinates, annotation], queue=False)
    clear.click(lambda: ("", None), outputs=[coordinates, annotation], queue=False)
    submit.click(match, [reference, current, variant, coordinates, confidence],
                 [connections, overlay, buttons, status, download], api_name="match", concurrency_limit=1)
    gr.Examples([[str(ROOT / "RoMa/assets/toronto_A.jpg"), str(ROOT / "RoMa/assets/toronto_B.jpg")]],
                inputs=[reference, current], label="官方示例：点击填入图片，然后开始匹配")

if __name__ == "__main__":
    import httpx
    import webbrowser
    try:
        existing = httpx.get("http://127.0.0.1:7860/config", timeout=2, trust_env=False)
        if existing.status_code == 200 and existing.json().get("title") == "RoMa 电梯面板匹配":
            webbrowser.open("http://127.0.0.1:7860")
            raise SystemExit(0)
    except httpx.RequestError:
        pass
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
                                                   share=False, inbrowser=True,
                                                   allowed_paths=[str(ROOT / "results"), str(ROOT / "RoMa/assets")])
