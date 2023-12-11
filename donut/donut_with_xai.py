import argparse
import gradio as gr
import torch
from donut import DonutModel
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image

def apply_fusion(fusion_mode: str, unfused_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if fusion_mode == "mean":
        fused_tensor = torch.mean(unfused_tensor, dim=dim)
    elif fusion_mode == "max":
        fused_tensor = torch.max(unfused_tensor, dim=dim)[0]
    elif fusion_mode == "min":
        fused_tensor = torch.min(unfused_tensor, dim=dim)[0]
    else:
        raise NotImplementedError(f"{fusion_mode} fusion not supported")
    return fused_tensor

def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt
    cv_img_resized = cv2.resize(input_img, (960, 1280))
    pil_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(pil_img, prompt=user_prompt, return_attentions=True)
    parsed_out = output["predictions"][0]
    
    cross_attentions = output["attentions"]["cross_attentions"]
    token_indices = [_ for _ in range(len(cross_attentions))]
    agg_heatmap = np.zeros([1280, 960], dtype=np.uint8)
    head_fusion_type = ["mean", "max", "min"][1]
    layer_fusion_type = ["mean", "max", "min"][1]
    for tidx in token_indices:    
        hmaps = torch.stack(cross_attentions[tidx], dim=0)
        
        # shape [4, 1, 16, 1, 4800]->[1, 4, 16, 4800]
        hmaps = hmaps.permute(1, 3, 0, 2, 4).squeeze(0)
        
        # shape [1, 4, 16, 4800]->[4, 16, 4800]
        hmaps = hmaps[-1]
        
        # change shape [4, 16, 4800]->[4, 16, 80, 60]
        hmaps = hmaps.view(4, 16, 80, 60)
        
        # fusing 16 decoder attention heads i.e. [4, 16, 80, 60]-> [16, 80, 60]
        hmaps = apply_fusion(head_fusion_type, hmaps, dim=1)
        
        # fusing 4 decoder layers from BART i.e. [16, 80, 60]-> [80, 60]
        hmap = apply_fusion(layer_fusion_type, hmaps, dim=0)
        hmap = hmap.unsqueeze(dim=-1).cpu().numpy()
        hmap = (hmap * 255.).astype(np.uint8)  # (80, 60, 1) uint8
        
        # fuse heatmaps for different tokens by taking the max
        agg_heatmap = np.max(np.asarray([agg_heatmap, cv2.resize(hmap, (960, 1280))]), axis=0).astype(np.uint8)
        
    raw_heatmap = deepcopy(agg_heatmap)
    raw_image = deepcopy(cv_img_resized)
    
    # threshold to remove small attention pockets
    thres_heatmap = cv2.threshold(agg_heatmap, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    contours = cv2.findContours(thres_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # heatmap_img = cv2.applyColorMap(thres_heatmap, cv2.COLORMAP_JET)
    super_imposed_raw_heatmap_img = cv2.addWeighted(cv2.applyColorMap(raw_heatmap, cv2.COLORMAP_JET), 0.5, raw_image, 0.5, 0)
    
    return super_imposed_raw_heatmap_img, parsed_out


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
    args, left_argv = parser.parse_known_args()


    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"


    pretrained_model = DonutModel.from_pretrained(args.pretrained_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)
    else:
        pretrained_model.encoder.to(torch.bfloat16)

    pretrained_model.eval()
    
    # for input (gradio)
    input_image = gr.components.Image(type="numpy", label="Document Image")
    input_text = gr.components.Textbox(lines=2, label="Question")
    
    # for output (gradio)
    cross_attention_map = gr.components.Image(type="numpy", label="Cross Attention Map")
    parsed_output = gr.components.JSON(label="Parsed Output")
    
    demo = gr.Interface(
        fn=demo_process_vqa,
        inputs=[input_image, input_text],
        outputs=[cross_attention_map, parsed_output],
        title=f"Donut 🍩 demonstration for Delivery Order Inference",
    )
    demo.launch()