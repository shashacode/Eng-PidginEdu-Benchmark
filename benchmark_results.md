# Eng-PidginEdu benchmark

MT scores are against **clean references** (glosses stripped from both sides), so they are comparable to standard English-Pidgin MT results.

| Model | Checkpoint | BLEU | chrF++ | TER | AfriCOMET | GlossAcc | GlossF1 |
|---|---|---|---|---|---|---|---|
| mt5_large | google/mt5-large | 66.92 | 80.59 | 32.23 | 71.94 | 79.68 | 78.51 |
| mt5_large | google/mt5-large | 66.51 | 79.65 | 33.45 | N/A | 79.58 | 78.50 |
| m2m100_1.2b | facebook/m2m100_1.2B | 68.50 | 80.14 | 30.81 | 71.76 | 77.15 | 77.20 |
| m2m100_1.2b | facebook/m2m100_1.2B | 54.27 | 78.01 | 37.11 | N/A | 76.56 | 76.77 |
| mbart50 | facebook/mbart-large-50-many-to-many-mmt | 68.18 | 79.96 | 30.82 | 71.80 | 75.90 | 76.74 |
| afriteva_v2_large | castorini/afriteva_v2_large | 66.32 | 79.57 | 32.85 | 71.62 | 78.58 | 76.54 |
| afriteva_v2_large | castorini/afriteva_v2_large | 65.95 | 79.27 | 33.37 | N/A | 77.53 | 76.26 |
| seamless | facebook/seamless-m4t-v2-large | 65.05 | 79.59 | 35.21 | 71.67 | 78.36 | 76.08 |
| nllb | facebook/nllb-200-distilled-600M | 63.47 | 78.73 | 36.99 | N/A | 80.03 | 75.99 |
| nllb | facebook/nllb-200-distilled-600M | 64.61 | 79.60 | 35.83 | 71.50 | 79.86 | 75.88 |
| mbart50 | facebook/mbart-large-50-many-to-many-mmt | 67.30 | 79.27 | 32.29 | N/A | 74.73 | 75.66 |
| seamless | facebook/seamless-m4t-v2-large | 67.56 | 79.74 | 32.51 | 71.92 | 75.75 | 75.52 |
| seamless | facebook/seamless-m4t-v2-large | 67.17 | 79.59 | 32.67 | N/A | 75.51 | 75.18 |
| mbart50 | facebook/mbart-large-50-many-to-many-mmt | 67.17 | 79.23 | 32.40 | 71.50 | 74.36 | 74.59 |
| m2m100 | facebook/m2m100_418M | 53.75 | 77.25 | 37.53 | N/A | 74.13 | 74.15 |
| m2m100_1.2b | facebook/m2m100_1.2B | 66.60 | 78.82 | 32.25 | 71.38 | 73.66 | 74.00 |
| toucan | UBC-NLP/toucan-1.2B | 64.74 | 79.34 | 35.70 | 71.80 | 75.35 | 73.96 |
| toucan | UBC-NLP/toucan-1.2B | 64.25 | 78.28 | 36.33 | N/A | 75.18 | 73.96 |
| cheetah | UBC-NLP/cheetah-1.2B | 64.91 | 78.48 | 35.87 | N/A | 74.58 | 73.84 |
| nllb | facebook/nllb-200-distilled-600M | 63.89 | 79.05 | 36.21 | 71.26 | 77.55 | 73.75 |
| m2m100 | facebook/m2m100_418M | 66.38 | 78.66 | 33.45 | 71.26 | 73.15 | 73.28 |
| cheetah | UBC-NLP/cheetah-1.2B | 66.16 | 79.62 | 33.80 | 71.84 | 74.14 | 73.16 |
| madlad3b | google/madlad400-3b-mt | 61.43 | 78.44 | 40.41 | 71.15 | 73.07 | 70.78 |
| m2m100 | facebook/m2m100_418M | 65.62 | 78.31 | 34.26 | 70.86 | 69.85 | 70.65 |
| t5v11xl | google/t5-v1_1-xl | 63.68 | 78.09 | 36.61 | 71.03 | 66.14 | 66.36 |
| toucan | UBC-NLP/toucan-1.2B | 64.38 | 78.28 | 36.07 | 71.17 | 62.47 | 64.18 |
| afriteva_v2_large | castorini/afriteva_v2_large | 65.87 | 78.59 | 34.03 | 71.15 | 59.35 | 62.50 |
| mt5_large | google/mt5-large | 63.32 | 77.46 | 38.23 | 70.74 | 58.88 | 61.68 |
| cheetah | UBC-NLP/cheetah-1.2B | 63.36 | 77.67 | 37.21 | 70.89 | 57.78 | 60.75 |
| afrimt5 | masakhane/afri-mt5-base | 63.41 | 78.19 | 37.88 | 70.82 | 56.09 | 60.32 |
| afrimt5 | masakhane/afri-mt5-base | 61.61 | 77.24 | 40.40 | N/A | 54.76 | 59.49 |
| mt5 | google/mt5-base | 62.65 | 77.79 | 38.87 | 70.63 | 50.11 | 56.11 |
| mt5 | google/mt5-base | 60.18 | 76.57 | 42.81 | N/A | 48.71 | 55.17 |
| afriteva | castorini/afriteva_base | 55.40 | 73.01 | 46.46 | N/A | 42.18 | 49.42 |
| afriteva | castorini/afriteva_base | 57.71 | 74.38 | 43.54 | 63.21 | 41.71 | 49.01 |
| afriteva | castorini/afriteva_base | 57.50 | 74.96 | 40.62 | 66.39 | 24.17 | 31.26 |
| afrimt5 | masakhane/afri-mt5-base | 63.56 | 77.61 | 37.89 | 70.37 | 20.51 | 29.22 |
| mt5 | google/mt5-base | 61.58 | 76.74 | 40.49 | 69.79 | 12.00 | 18.52 |
| nllb | facebook/nllb-200-distilled-600M | 2.08 | 14.20 | 150.65 | 23.51 | 0.04 | 0.07 |
| afrimt5 | masakhane/afri-mt5-base | 0.03 | 4.31 | 273.86 | 21.84 | 0.00 | 0.00 |
| afriteva_v2_large | castorini/afriteva_v2_large | 5.84 | 19.42 | 110.28 | 21.74 | 0.00 | 0.00 |
| cheetah | UBC-NLP/cheetah-1.2B | 0.00 | 0.01 | 99.99 | 15.85 | 0.00 | 0.00 |
| m2m100 | facebook/m2m100_418M | 27.08 | 57.93 | 65.64 | 58.33 | 0.00 | 0.00 |
| afriteva | castorini/afriteva_base | 0.30 | 4.82 | 100.96 | 16.42 | 0.00 | 0.00 |
| m2m100_1.2b | facebook/m2m100_1.2B | 28.23 | 58.85 | 63.26 | 59.42 | 0.00 | 0.00 |
| madlad3b | google/madlad400-3b-mt | 5.80 | 25.96 | 188.69 | 32.16 | 0.00 | 0.00 |
| mt5 | google/mt5-base | 0.28 | 4.88 | 98.41 | 18.39 | 0.00 | 0.00 |
| mbart50 | facebook/mbart-large-50-many-to-many-mmt | 2.34 | 10.19 | 109.06 | 41.24 | 0.00 | 0.00 |
| mt5_large | google/mt5-large | 0.97 | 15.83 | 304.91 | 25.70 | 0.00 | 0.00 |
| seamless | facebook/seamless-m4t-v2-large | 32.71 | 62.53 | 59.64 | 58.55 | 0.00 | 0.00 |
| t5v11xl | google/t5-v1_1-xl | 3.16 | 11.88 | 109.98 | 22.09 | 0.00 | 0.00 |
| toucan | UBC-NLP/toucan-1.2B | 17.01 | 40.68 | 90.34 | 50.47 | 0.00 | 0.00 |
