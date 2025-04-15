## Progressive Complexity in Synthetic Datasets

| Dataset | Description | Sample Images & Labels |
|---------|------------|------------------------|
| **Dataset I** | White Lines with Varied Angles | [View Samples](https://huggingface.co/datasets/nubahador/LinePropsDataset/tree/main) |
| **Dataset II** | White Lines with Varied Angles, Lengths | |
| **Dataset III** | White Lines with Varied Angles, Lengths, and Widths | |
| **Dataset IV** | Lines with Varied Angles, Lengths, Widths, and Colors | |

---

## Fine-tuned Models

<div style="border-left: 3px solid #0366d6; padding-left: 15px; margin-bottom: 20px;">

1. **Model I**: Fine-tuned on White Lines with Varied Angles  
   [View Model](https://huggingface.co/nubahador/FT-Transformer-LineProps/tree/main/white_lines_varying_angles_fine_tuned_model)

2. **Model II**: Fine-tuned on White Lines with Varied Angles, Lengths  
   [View Model](https://huggingface.co/nubahador/FT-Transformer-LineProps/tree/main/white_lines_varying_angles_lengths_fine_tuned_model)

3. **Model III**: Fine-tuned on White Lines with Varied Angles, Lengths, and Widths  
   [View Model](https://huggingface.co/nubahador/FT-Transformer-LineProps/tree/main/white_lines_with_varying_angles_lengths_and_widths_fine_tuned_model)

4. **Model IV**: Fine-tuned on Lines with Varied Angles, Lengths, Widths, and Colors  
   [View Model](https://huggingface.co/nubahador/FT-Transformer-LineProps/tree/main/white_lines_with_varying_angles_lengths_widths_colors_fine_tuned_model)

</div>

---

### Visual Overview

### Sample Generated Images
<div align="center">
  <img src="https://github.com/nbahador/Vision_Transformers_Exhibit_Human_Like_Biases/blob/main/Figures/Sample_generated_images.png" alt="Sample generated images for all datasets" style="max-width: 80%; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
  <p style="font-size: 0.9em; color: #666;">Figure 1: Sample images from all four synthetic datasets showing progressive complexity</p>
</div>

### Model Architecture
<div align="center">
  <img src="https://github.com/nbahador/Vision_Transformers_Exhibit_Human_Like_Biases/blob/main/Figures/Fine-tuned_on_Lines_with_Varied_Angles_Lengths_Widths_and_Colors.png" alt="Architecture diagram of Fine-Tuned Model on Dataset IV" style="max-width: 70%; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
  <p style="font-size: 0.9em; color: #666;">Figure 2: Architecture of the model fine-tuned on Dataset IV (most complex variant)</p>
</div>

---
---

## Citation

[Bahador N. Vision Transformers exhibit human-like biases: Evidence of orientation and color selectivity, categorical perception, and phase transitions. arXiv [csCV]. Published online 2025. doi:10.48550/ARXIV.2504.09393
 	 ](https://arxiv.org/pdf/2504.09393)

<div style="background-color: #f6f8fa; padding: 15px; border-radius: 6px; border-left: 4px solid #2188ff; margin: 20px 0;">

```bibtex
@article{bahador2025vision,
  title={Vision Transformers exhibit human-like biases: Evidence of orientation and color selectivity, categorical perception, and phase transitions},
  author={Bahador, N},
  journal={arXiv [cs.CV]},
  year={2025},
  doi={10.48550/ARXIV.2504.09393},
  url={https://arxiv.org/pdf/2504.09393}
}
