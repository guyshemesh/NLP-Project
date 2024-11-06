FrankenModels Project:

This project investigates the impact of duplicating frozen transformer layers in large language models (LLMs) like Pythia and LLaMA3. Our primary goal is to determine whether duplicating multi-head self-attention (MHSA), without additional training, can enhance performance on complex NLP tasks. To this end, we conducted extensive evaluations on custom models of pythia-70m-deduped, created by duplicating selected MHSA layers, and tested them across a wide range of BIG-bench tasks. Additionally, limited evaluations of larger models, pythia-6.9b and llama-3-8B, provided perspective on the approach’s viability for larger models.

This study outlines the motivations, experimental design, results, and avenues for future work. Our findings offer initial insights into layer duplication as a cost-effective strategy for improving LLM performance without retraining. Specifically, we discover that duplicating certain layers a certain amount of times in pythia-70m-deduped yield up to a 15 percent relative performance increase.

group members:
Neo Eyal neoedan@gmail.com
Adi Shani adishani1@mail.tau.ac.il
Milana Yakubov milanay1@mail.tau.ac.il
Guy Shemesh Guyshemesh@mail.tau.ac.il

Mentor : Dr. Mor Geva
