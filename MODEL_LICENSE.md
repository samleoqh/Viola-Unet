# Viola v3.1 Model License

Copyright (c) 2023–2025 Qinghui Liu and Contributors

---

## 1. Scope

This license applies to the pre-trained model weights and model artifacts of Viola v3.1, including but not limited to:

- Neural network parameter files (`.pth`, `.pt`, `.ckpt`, `.safetensors`, `.onnx`, and any equivalent formats)
- Packaged Docker images containing pre-trained weights
- Standalone GUI applications that embed pre-trained weights
- Any inference servers or deployment packages bundling the above weights

It does **not** apply to the source code of Viola-UNet, which is licensed separately under the [Apache License 2.0](LICENSE).

## 2. License Terms

The pre-trained weights and model artifacts are licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** (CC BY-NC-ND 4.0) license.

Full legal text: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode

### Summary of your obligations and restrictions:

**You are free to:**
- Download and use the weights for academic research, education, and non-commercial evaluation.
- Cite the original work in publications.

**You must:**
- Provide appropriate credit to the original authors (Qinghui Liu et al.) and the datasets used for training.
- Include a link to this license.

**You may NOT:**
- Use the pre-trained weights or model artifacts for any commercial purpose, including but not limited to:
  - Integration into commercial software or medical devices.
  - Use in clinical workflows that generate revenue.
  - Redistribution as part of a paid product or service.
- Create and distribute derivative works based on these weights (e.g., fine-tuned versions, distilled models, quantized models, LoRA/PEFT adapters, or modified ensembles) without explicit written permission.
- Remove or alter attribution notices.

## 3. Why These Restrictions Exist

The pre-trained models were trained on publicly available datasets licensed under CC BY-NC-ND 4.0:

- **[INSTANCE Challenge 2022](https://instance.grand-challenge.org/)** — Intracranial Hemorrhage Segmentation dataset
- **[BHSD Dataset](https://github.com/White65534/BHSD)** — Brain Hemorrhage Segmentation Dataset

Because these datasets explicitly prohibit commercial use and the creation of derivative works, we are legally bound to pass these same restrictions on to any downstream user of the pre-trained weights. **We cannot grant exceptions or alternative commercial licenses for the weights themselves.**

## 4. Commercial Use Path

If you wish to use Viola-UNet for commercial purposes, you are welcome to use the source code (licensed under Apache 2.0) and retrain the models on your own commercially-licensed data.

## 5. Disclaimer

THE PRE-TRAINED WEIGHTS AND MODEL ARTIFACTS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE WEIGHTS OR THE USE OR OTHER DEALINGS IN THE WEIGHTS.

These models are intended for research and educational purposes only. They are **not** FDA/CE-approved medical devices and must not be used as the sole basis for clinical diagnosis or treatment decisions.
