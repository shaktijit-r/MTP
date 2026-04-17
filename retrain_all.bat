@echo off
echo ================================================================
echo   FULL RETRAINING PIPELINE v3 (Speed + Accuracy Optimized)
echo   - 80K sample cap per epoch (full data access, faster epochs)
echo   - Batch 32 (12GB VRAM safe), Patience 5, Max 25 epochs
echo   - Class-aware weighted FocalLoss + EMA + TTA
echo   - Estimated: ~48-60 hours on RTX 4080
echo ================================================================
echo.

set PYTHONPATH=l:\MTP;l:\MTP\src\data;l:\MTP\src
cd /d l:\MTP

echo [Phase 1/4] Cross-Dataset Merged Training (9 diseases)
echo ================================================================
python train_merged.py
if %ERRORLEVEL% NEQ 0 echo WARNING: Merged training had errors!
echo.

echo [Phase 2/4] Single-Source NIH Diseases (7 diseases)
echo ================================================================

echo --- Emphysema ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Emphysema
echo.

echo --- Fibrosis ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Fibrosis
echo.

echo --- Hernia ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Hernia
echo.

echo --- Infiltration ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Infiltration
echo.

echo --- Mass ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Mass
echo.

echo --- Nodule ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Nodule
echo.

echo --- Pleural_Thickening ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\NIH" --target_disease Pleural_Thickening
echo.

echo [Phase 3/4] Single-Source MIMIC + COVID + Ophthalmology + Dermatology + Neurology
echo ================================================================

echo --- Enlarged_Cardiomediastinum (MIMIC) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\MIMIC" --target_disease Enlarged_Cardiomediastinum
echo.

echo --- Fracture (MIMIC) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\MIMIC" --target_disease Fracture
echo.

echo --- Lung_Lesion (MIMIC) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\MIMIC" --target_disease Lung_Lesion
echo.

echo --- COVID (covid-19) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\covid-19" --target_disease COVID
echo.

echo --- Viral_Pneumonia (covid-19) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\chestxray\covid-19" --target_disease Viral_Pneumonia
echo.

echo --- AMD (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease AMD
echo.

echo --- Cataract (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease Cataract
echo.

echo --- Diabetes (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease Diabetes
echo.

echo --- Glaucoma (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease Glaucoma
echo.

echo --- Hypertension (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease Hypertension
echo.

echo --- Myopia (ODIR) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\ODIR" --target_disease Myopia
echo.

echo --- Diabetic_Retinopathy ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\ophthalmology\diabetic-retinopathy-detection" --target_disease Diabetic_Retinopathy
echo.

echo --- akiec (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease akiec
echo.

echo --- bcc (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease bcc
echo.

echo --- bkl (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease bkl
echo.

echo --- df (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease df
echo.

echo --- nv (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease nv
echo.

echo --- vasc (HAM10000) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\dermatology\HAM10000" --target_disease vasc
echo.

echo --- Dementia (OASIS) ---
python src\models\train_baseline.py --domain_path "l:\MTP\datasets\neurology\OASIS" --target_disease Dementia
echo.

echo [Phase 4/4] Export All Models to Mobile
echo ================================================================
python export_mobile.py

echo.
echo ================================================================
echo   PIPELINE COMPLETE! All 35 models trained and exported.
echo   Check mobile_app\assets\ for .ptl files.
echo ================================================================
pause
