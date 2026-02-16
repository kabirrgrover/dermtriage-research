# Skin Lesion Triage: AI-Powered Clinical Decision Support

A clinical decision support tool that helps primary care physicians identify potentially dangerous skin lesions using smartphone photos. Built with Google's MedSigLIP and MedGemma medical AI foundation models, with a focus on fairness across skin tones.

**Status:** Research prototype with validated cross-domain performance

---

## The Problem

Skin cancer is the most common cancer in the United States, affecting over 5 million Americans annually. Melanoma, the deadliest form, kills approximately 8,000 people each year. Early detection is critical — the five-year survival rate for melanoma caught early exceeds 99%, but drops to around 30% once the cancer has spread.

Two interconnected problems make early detection difficult:

**1. Access Gap.** Primary care physicians are often the first to see patients with skin concerns, but they achieve only about 45% diagnostic accuracy for melanoma compared to 97% for dermatologists. Dermatologists are in short supply, particularly in rural and underserved communities. Patients in these areas may wait weeks or months for specialist appointments.

**2. Fairness Gap.** Most AI tools for skin cancer detection are trained on datasets composed predominantly of lighter skin tones (Fitzpatrick I-III). As a result, these tools perform significantly worse on patients with darker skin, compounding existing health disparities.

This project addresses both problems by building an AI tool that works on clinical smartphone images and is explicitly trained and evaluated to perform well across different skin tones.

---

## Approach

### Dual-Model Architecture

| Model | Purpose |
|-------|---------|
| **MedSigLIP-448** | Vision encoder for lesion classification (frozen) |
| **Classification Head** | 7-class skin lesion classifier (trainable) |
| **MedGemma-4B** | Natural language clinical explanations |

### Seven-Class Classification

| Class | Description | Risk Level |
|-------|-------------|------------|
| mel | Melanoma | URGENT — Refer within 48 hours |
| bcc | Basal cell carcinoma | HIGH — Refer within 2 weeks |
| akiec | Actinic keratosis | MODERATE — Refer within 2-4 weeks |
| bkl | Benign keratosis | LOW — Routine monitoring |
| df | Dermatofibroma | LOW — Routine monitoring |
| nv | Melanocytic nevus | LOW — Routine monitoring |
| vasc | Vascular lesion | LOW — Routine monitoring |

### Training Strategy

- **Loss Function:** Focal Loss (γ=2.0) to focus on hard-to-classify examples
- **Class Weighting:** 2x weight on melanoma to prioritize recall
- **Mixed Training:** Combined dermoscopic (HAM10000) and clinical smartphone images (PAD-UFES-20)
- **Augmentation:** ColorJitter and GaussianBlur to simulate real-world smartphone variability

---

## Datasets

### HAM10000
- 10,015 dermoscopic images from specialized clinical equipment
- 7 lesion classes
- Fitzpatrick skin types predominantly I-III (lighter skin tones)

### PAD-UFES-20
- ~2,100 clinical smartphone images
- Brazilian population with Fitzpatrick types I-V
- Real-world image quality (variable lighting, blur, angles)

---

## Results

### Experiment 1: Baseline (HAM10000 Only)

Trained on dermoscopic images, evaluated on both image types.

| Dataset | Balanced Accuracy | Melanoma Recall |
|---------|-------------------|-----------------|
| HAM10000 (dermoscopic) | 78.9% | 85.7% |
| PAD-UFES-20 (clinical) | 49.9% | 54.5% |

**Finding:** ~30% performance drop on clinical smartphone images. A model trained only on dermoscopic images fails to generalize to the real-world images primary care physicians would capture.

### Experiment 2: Mixed Training (HAM10000 + PAD-UFES-20)

Trained on both datasets with 3x oversampling of clinical images.

| Dataset | Balanced Accuracy | Melanoma Recall | Change |
|---------|-------------------|-----------------|--------|
| HAM10000 | 69.6% | 77.6% | -9.3% / -8.1% |
| PAD-UFES-20 | **81.8%** | **81.8%** | **+31.9% / +27.3%** |

**Finding:** Mixed training bridges the domain gap. Clinical image melanoma recall improved from 54.5% to 81.8% (+27 points). The trade-off is acceptable given the target deployment (primary care with smartphones).

### Domain Gap Reduction

| Metric | Baseline Gap | After Mixed Training | Reduction |
|--------|--------------|---------------------|-----------|
| Balanced Accuracy | 29.0 points | 12.2 points | 58% smaller |
| Melanoma Recall | 31.2 points | 4.2 points | **87% smaller** |

### Fairness Analysis (Clinical Images)

| Model | Fitzpatrick Type | n | Accuracy | Melanoma Recall |
|-------|------------------|---|----------|-----------------|
| Baseline | I-II | 168 | 51.2% | 50.0% |
| Baseline | III-IV | 93 | 46.2% | 66.7% |
| Mixed | I-II | 168 | 86.3% | 75.0% |
| Mixed | III-IV | 93 | 82.8% | **100.0%** |

Mixed training improved melanoma recall for both skin tone groups, with Fitzpatrick III-IV achieving 100% recall.

---

## Explainability Features

- **Grad-CAM:** Visual heatmaps showing which image regions drive predictions
- **Monte Carlo Dropout:** Uncertainty quantification through multiple forward passes
- **Temperature Scaling:** Calibrated confidence scores
- **MedGemma Explanations:** Natural language clinical assessments explaining predictions

---

## Known Issues and Concerns

### 1. Unknown Fitzpatrick Group
The "Unknown" Fitzpatrick group in PAD-UFES-20 shows 0% melanoma recall. This needs investigation — either there are few melanoma samples in this subgroup, or the model systematically fails on them.

### 2. Calibration Not Verified
Confidence scores have not been calibrated on the mixed model. Overconfident predictions could be harmful in clinical settings.

### 3. Limited Clinical Dataset
PAD-UFES-20 is a single clinical dataset from Brazil. Performance on images from other populations, phone cameras, or lighting conditions is unknown.

### 4. No Fitzpatrick V-VI Evaluation
Neither dataset contains sufficient samples from the darkest skin tones (Fitzpatrick V-VI). Performance on these populations remains untested.

### 5. Domain Trade-off
The mixed model improves clinical image performance but reduces dermoscopic performance by ~9 points. Different deployment settings may require different models.

### 6. Retrospective Validation Only
All results are from retrospective validation on research datasets. Prospective clinical validation has not been performed.

---

## Future Directions

1. **Calibration:** Run temperature scaling and report Expected Calibration Error
2. **Additional Datasets:** Evaluate on DDI (balanced Fitzpatrick I-VI) for fairness
3. **Clinical Pilot:** Test with primary care clinicians for real-world feedback
4. **Prospective Validation:** Design study for real-world deployment evaluation

---

## Technical Details

- **Vision Encoder:** MedSigLIP-448 (frozen during training)
- **Input Size:** 448×448 pixels
- **Training:** Google Colab with A100 GPU
- **Framework:** PyTorch + HuggingFace Transformers

---

## References

1. Tschandl P, et al. "The HAM10000 dataset." *Scientific Data*, 2018.
2. Pacheco AG, et al. "PAD-UFES-20: A skin lesion dataset." *Data in Brief*, 2020.
3. Daneshjou R, et al. "Disparities in dermatology AI performance." *Science Advances*, 2022.

---

## Disclaimer

**This project is for research and education only.**

- This is NOT a medical device and must not be used for clinical decisions
- Results are from retrospective validation on research datasets
- Prospective clinical validation has not been performed
- Known limitations include lack of Fitzpatrick V-VI evaluation
- Always consult qualified healthcare professionals for medical decisions

---

*Last Updated: February 2026*
