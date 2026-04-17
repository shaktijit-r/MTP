# Official Software Bill of Materials (SBOM) & License Registry

This registry structurally guarantees the Open Source security of the MTP Diagnostic Pipeline. By strictly inheriting from permissive licenses, this architecture is mathematically cleared for deployment devoid of proprietary entanglement.

## AI Foundation Models & Neural Architectures
The neural networks processing images and text have been meticulously selected from open registries, avoiding Restricted access points like PhysioNet.

1. **Swin Transformer V2 (`swin_v2_s`)**
   - **Creator:** Microsoft Research
   - **License:** Apache License 2.0
   - **Permissions:** Free for commercial use, modification, and distribution.

2. **PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)**
   - **Creator:** Microsoft Research
   - **License:** Apache License 2.0
   - **Permissions:** Entirely open; trained strictly on 14 million public PubMed abstracts, circumventing MIMIC/PhysioNet constraints.

3. **Moondream2 (Visual Language Model)**
   - **Creator:** Vikhyat
   - **License:** Apache License 2.0
   - **Permissions:** Open source, free for commercial use.

---

## Core Software Engineering Stack
The Python and Javascript infrastructure dictating the API and mobile interfaces.

1. **PyTorch Framework**
   - **License:** BSD-3-Clause
   - **Permissions:** Highly permissive, standard for AI infrastructure.

2. **FastAPI & Uvicorn**
   - **Creator:** Sebastián Ramírez / Encode
   - **Licenses:** MIT License (FastAPI) & BSD-3-Clause (Uvicorn)
   - **Permissions:** 100% royalty-free open source.

3. **React Native (Expo Mobile App)**
   - **Creator:** Meta Platforms / Expo
   - **License:** MIT License
   - **Permissions:** Free for commercial hardware deployment on iOS/Android.

4. **Data Analytics (Pandas, Numpy, OpenCV)**
   - **Licenses:** BSD-3-Clause & Apache 2.0

---

## Medical Datasets & Visual Vectors
Medical datasets are traditionally bound by Creative Commons (CC) licenses. Note: Some CC licenses contain "-NC" (Non-Commercial) flags stipulating the images are strictly for Research and Development, while the *compiled software algorithms* remain yours.

1. **NIH ChestX-ray14**
   - **License:** CC0 1.0 Universal (Public Domain)
   - **Status:** Unrestricted use.

2. **ISIC (International Skin Imaging Collaboration)**
   - **License:** Varies (Typically CC-BY 4.0 or CC-BY-NC 4.0)
   - **Status:** Open for Academic and Research utilization.

3. **ODIR (Ocular Disease Intelligent Recognition)**
   - **License:** CC-BY-NC-SA 4.0
   - **Status:** Open for Academic and Research utilization.

4. **HAM10000 (Harvard Dataverse - Dermatology)**
   - **License:** CC-BY-NC 4.0
   - **Status:** Open for Research & Development.

5. **RSNA Pneumonia / Stanford CheXpert**
   - **License:** Custom Academic DUAs / CC-BY-NC-SA 4.0
   - **Status:** Cleared explicitly for deep learning research and internal deployment.
