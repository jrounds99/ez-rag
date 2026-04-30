# RAG comparison — summary

## L1 qwen2.5:3b

- Overall correct (RAG vs RAW): **9/10**  vs  **6/10**
- Corpus-question correct: **5/6**  vs  **2/6**
- Avg latency: RAG **1.3s**, RAW **0.7s**

| Q | cat | mode | ok | latency | answer head | issues |
|---|---|---|:---:|---:|---|---|
| Q1 | corpus | RAG | ✓ | 2.2s | The very last person to walk on the Moon was Eugene Cernan, and this occurred du | - |
| Q1 | corpus | RAW | ✓ | 0.9s | The very last person to walk on the Moon was Eugene Cernanov, though he did not  | - |
| Q2 | corpus | RAG | ✓ | 1.7s | Based on the information provided in your notes, the Border Collie is described  | - |
| Q2 | corpus | RAW | ✗ | 0.8s | I'm sorry, but I don't have access to your notes or any previous context from ou | missing-required:['border collie'] |
| Q3 | corpus | RAG | ✓ | 0.9s | Based on the information provided in the context items, the price of SKU A101 in | - |
| Q3 | corpus | RAW | ✗ | 0.8s | I'm sorry, but I don't have access to your store's specific inventory informatio | missing-required:['24.95'] |
| Q4 | corpus | RAG | ✗ | 1.3s | Based on the information provided in the context items, there is no mention of a | missing-required:['phoenix falcon'] |
| Q4 | corpus | RAW | ✗ | 0.8s | I'm sorry, but I don't have access to any OCR test screenshot or previous conver | missing-required:['phoenix falcon'] |
| Q5 | corpus | RAG | ✓ | 1.3s | According to the context item [1], the Pacific Ocean covers about 165 million sq | - |
| Q5 | corpus | RAW | ✗ | 0.7s | According to my knowledge, the Pacific Ocean covers approximately 185.2 million  | missing-required:['165'] |
| Q6 | corpus | RAG | ✓ | 1.0s | The pigment in chloroplasts that captures light energy is chlorophyll, as mentio | - |
| Q6 | corpus | RAW | ✓ | 0.8s | The pigment in chloroplasts that captures light energy is chlorophyll. Specifica | - |
| Q7 | general | RAG | ✓ | 1.0s | The capital of Japan is Tokyo. According to context item [1], it discusses atten | - |
| Q7 | general | RAW | ✓ | 0.6s | The capital of Japan is Tokyo. | - |
| Q8 | general | RAG | ✓ | 1.2s | HTTP stands for HyperText Transfer Protocol. This is a standard protocol used fo | - |
| Q8 | general | RAW | ✓ | 0.6s | HTTP stands for Hypertext Transfer Protocol. | - |
| Q9 | general | RAG | ✓ | 1.0s | The speed of light in a vacuum is approximately 299,792 kilometers per second. T | - |
| Q9 | general | RAW | ✓ | 0.7s | The speed of light in a vacuum is approximately 299,792 kilometers per second. | - |
| Q10 | general | RAG | ✓ | 1.0s | Photosynthesis is a process by which green plants use sunlight to synthesize foo | - |
| Q10 | general | RAW | ✓ | 0.7s | Photosynthesis is the process by which plants use sunlight, water, and carbon di | - |

### Issues found
- **Q2 RAW**: missing-required:['border collie']
- **Q3 RAW**: missing-required:['24.95']
- **Q4 RAG**: missing-required:['phoenix falcon']
- **Q4 RAW**: missing-required:['phoenix falcon']
- **Q5 RAW**: missing-required:['165']

## L2 deepseek-r1:32b

- Overall correct (RAG vs RAW): **10/10**  vs  **8/10**
- Corpus-question correct: **6/6**  vs  **4/6**
- Avg latency: RAG **7.3s**, RAW **5.2s**

| Q | cat | mode | ok | latency | answer head | issues |
|---|---|---|:---:|---:|---|---|
| Q1 | corpus | RAG | ✓ | 7.7s | The last person to walk on the Moon was Eugene Cernan during the Apollo 17 missi | - |
| Q1 | corpus | RAW | ✓ | 6.8s | The very last person to walk on the Moon was Eugene Cernan during the Apollo 17  | - |
| Q2 | corpus | RAG | ✓ | 7.2s | The Border Collie is described as the most intelligent dog breed in the provided | - |
| Q2 | corpus | RAW | ✓ | 6.4s | I don’t have access to your personal notes or documents, so I can’t tell you whi | - |
| Q3 | corpus | RAG | ✓ | 5.8s | The price of SKU A101 in our store inventory is **$24.95** according to the pric | - |
| Q3 | corpus | RAW | ✗ | 3.9s | I don’t have access to real-time or internal company data, including specific pr | missing-required:['24.95'] |
| Q4 | corpus | RAG | ✓ | 8.6s | The magic phrase from the OCR test screenshot is "phoenix falcon" as stated in t | - |
| Q4 | corpus | RAW | ✗ | 5.7s | I don’t have access to any specific screenshots or documents, including OCR test | missing-required:['phoenix falcon'] |
| Q5 | corpus | RAG | ✓ | 7.6s | The Pacific Ocean covers approximately **165 million square kilometers** accordi | - |
| Q5 | corpus | RAW | ✓ | 7.7s | The Pacific Ocean is generally estimated to cover approximately 165 million squa | - |
| Q6 | corpus | RAG | ✓ | 7.0s | The pigment in chloroplasts that captures light energy is **chlorophyll**. This  | - |
| Q6 | corpus | RAW | ✓ | 4.4s | The pigment in chloroplasts that captures light energy is **chlorophyll**, speci | - |
| Q7 | general | RAG | ✓ | 10.8s | The capital of Japan is Tokyo. While the provided context documents focus on att | - |
| Q7 | general | RAW | ✓ | 2.9s | The capital of Japan is Tokyo. | - |
| Q8 | general | RAG | ✓ | 6.0s | HTTP stands for HyperText Transfer Protocol. It is the protocol used for transmi | - |
| Q8 | general | RAW | ✓ | 4.2s | HTTP stands for **HyperText Transfer Protocol**. It is the standard protocol use | - |
| Q9 | general | RAG | ✓ | 5.9s | The speed of light in a vacuum is approximately **299,792 kilometers per second* | - |
| Q9 | general | RAW | ✓ | 6.2s | The speed of light in a vacuum is approximately **299,792 kilometers per second* | - |
| Q10 | general | RAG | ✓ | 6.0s | Photosynthesis is the process by which green plants use sunlight to convert carb | - |
| Q10 | general | RAW | ✓ | 3.8s | Photosynthesis is the process by which plants, algae, and some bacteria convert  | - |

### Issues found
- **Q3 RAW**: missing-required:['24.95']
- **Q4 RAW**: missing-required:['phoenix falcon']
